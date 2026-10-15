export default {
  scheduled(controller, env, ctx) {
    ctx.waitUntil(handleScheduledTask(env));
  },
};

async function handleScheduledTask(env) {
  // 1. Timezone-aware check for 6 AM, 9 AM, 12 PM (Noon), 3 PM, and 6 PM Pacific Time
  const now = new Date();
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Los_Angeles",
    hour: "numeric",
    minute: "numeric",
    hourCycle: "h23",
  });

  const parts = formatter.formatToParts(now);
  const hour = parseInt(parts.find((p) => p.type === "hour").value, 10);
  const minute = parseInt(parts.find((p) => p.type === "minute").value, 10);

  // Matches 06:00, 09:00, 12:00, 15:00, and 18:00 Pacific within the first 10 mins
  const targetHours = [6, 9, 12, 15, 18];
  const isScheduledTime = targetHours.includes(hour) && minute < 10;

  if (isScheduledTime) {
    console.log(`Scheduled time trigger matched (${hour}:${minute.toString().padStart(2, "0")} PT). Triggering GitHub Action.`);
    await triggerGitHubAction(env, "Regularly scheduled deep analysis");
    return;
  }

  // 2. Fetch the RotoWire Fantasy NFL RSS Feed
  const feedUrl = "https://www.rotowire.com/rss/news.php?sport=NFL";
  let items = [];

  try {
    const res = await fetch(feedUrl, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Cloudflare-Worker-Fantasy-Manager",
        "Accept": "application/rss+xml, application/xml, text/xml",
      },
    });

    if (res.ok) {
      const xmlText = await res.text();
      console.log(`Fetched RotoWire Feed | XML Length: ${xmlText.length}`);
      items = parseRotoWireFeed(xmlText);
    } else {
      console.error(`Failed to fetch RotoWire feed. Status: ${res.status}`);
    }
  } catch (err) {
    console.error(`Error fetching RotoWire RSS feed: ${feedUrl}`, err);
  }

  if (items.length === 0) {
    console.log("No RSS items parsed from RotoWire feed.");
    return;
  }

  // 3. Filter for items that do NOT exist in KV yet
  const unseenItems = [];

  for (const item of items) {
    const kvKey = `seen:${item.guid}`;
    const alreadySeen = await env.SEEN_KV.get(kvKey);

    if (!alreadySeen) {
      unseenItems.push(item);
    }
  }

  // 4. If any unseen items exist, write ONLY the unseen GUIDs to KV and trigger GH Action
  if (unseenItems.length > 0) {
    const firstTitle = unseenItems[0].title || "New RotoWire Story";
    console.log(`Detected ${unseenItems.length} new story/stories. Writing only unseen GUIDs to KV and dispatching trigger.`);

    // Write only the unseen GUID keys to KV with a 7-day TTL (604,800 seconds)
    await Promise.all(
      unseenItems.map((item) =>
        env.SEEN_KV.put(`seen:${item.guid}`, "true", { expirationTtl: 604800 })
      )
    );

    // Trigger GitHub Action with the first unseen story title as the reason
    await triggerGitHubAction(env, firstTitle);
  } else {
    console.log("RotoWire feed checked: No new stories found.");
  }
}

/**
 * Triggers a GitHub Actions repository_dispatch event
 */
async function triggerGitHubAction(env, reason) {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/dispatches`;

  console.log(`Dispatching to URL: ${url}`);

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
      "Accept": "application/vnd.github.v3+json",
      "User-Agent": "Cloudflare-Worker",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      event_type: "manager-trigger",
      client_payload: { reason },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Failed to trigger GitHub Action: ${response.status} - ${errorText}`);
  } else {
    console.log(`Successfully dispatched GitHub Action (Reason: "${reason}").`);
  }
}

/**
 * Parses RSS <item> blocks to extract guid, title, and link
 */
function parseRotoWireFeed(xmlText) {
  const items = [];
  const itemRegex = /<item>([\s\S]*?)<\/item>/gi;
  let match;

  while ((match = itemRegex.exec(xmlText)) !== null) {
    const itemBlock = match[1];

    // RotoWire item tags extraction
    const guid = extractTag(itemBlock, "guid") || extractTag(itemBlock, "link");
    const title = extractTag(itemBlock, "title");
    const link = extractTag(itemBlock, "link");

    if (guid) {
      items.push({ guid, title, link });
    }
  }

  return items;
}

function extractTag(xml, tag) {
  const match = xml.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return match ? match[1].replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1").trim() : "";
}
