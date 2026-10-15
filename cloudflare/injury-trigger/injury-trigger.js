export default {
  scheduled(controller, env, ctx) {
    ctx.waitUntil(handleScheduledTask(env));
  },
};

async function handleScheduledTask(env) {
  console.log(`Triggering GitHub Action.`);
  await triggerGitHubAction(env);
  return;
}

/**
 * Triggers a GitHub Actions repository_dispatch event
 */
async function triggerGitHubAction(env) {
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
      event_type: "injury-trigger",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Failed to trigger GitHub Action: ${response.status} - ${errorText}`);
  } else {
    console.log(`Successfully dispatched GitHub Action.`);
  }
}
