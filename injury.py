import os
from datetime import datetime
import numpy as np
import pandas as pd
import nflreadpy as nfl
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import warnings

# Suppress verbose pandas and statsmodels warnings
warnings.filterwarnings('ignore')

CURRENT_YEAR = datetime.now().year
OUTPUT_PATH = "./data/injury.csv"

def get_most_recent_completed_regular_season_week():
    # Include CURRENT_YEAR in the search range (+1 for Python range upper-bound)
    seasons = list(range(CURRENT_YEAR - 3, CURRENT_YEAR + 1))
    schedules = nfl.load_schedules(seasons=seasons).to_pandas()

    # Filter for completed regular season games
    reg_games = schedules[(schedules['game_type'] == 'REG') & (schedules['result'].notnull())]

    latest_season = int(reg_games['season'].max())
    latest_week = int(reg_games[reg_games['season'] == latest_season]['week'].max())

    return latest_season, latest_week


class NFLInjuryPredictor:
    def __init__(self):
        self.end_season, self.end_week = get_most_recent_completed_regular_season_week()
        self.start_season = self.end_season - 7 # go back 7 years
        self.features = [
            'age', 'snaps_4wk_avg', 'career_injuries', 'weight', 
            'is_LP', 'is_DNP'
        ]

    def _load_nflverse_data(self):
        """
        Pulls data directly via nflreadpy and converts Polars DataFrames to Pandas.
        """
        seasons = list(range(self.start_season, self.end_season + 1))

        print("Downloading injury data...")
        self.injuries = nfl.load_injuries(seasons=seasons).to_pandas()

        print("Downloading weekly roster data...")
        self.rosters = nfl.load_rosters_weekly(seasons=seasons).to_pandas()

        print("Downloading snap counts...")
        self.snaps = nfl.load_snap_counts(seasons=seasons).to_pandas()

    def _engineer_features(self):
        """
        Merges datasets and engineers rolling weekly features.
        """
        print("Engineering features...")

        # 1. Calculate player age from birth_date if 'age' isn't explicitly present
        if 'age' not in self.rosters.columns:
            if 'birth_date' in self.rosters.columns:
                self.rosters['birth_date'] = pd.to_datetime(self.rosters['birth_date'], errors='coerce')
                self.rosters['age'] = self.rosters['season'] - self.rosters['birth_date'].dt.year
            else:
                self.rosters['age'] = np.nan

        if 'weight' not in self.rosters.columns:
            self.rosters['weight'] = np.nan

        # 2. Clean and aggregate snap counts
        snap_cols = ['pfr_player_id', 'season', 'week', 'offense_snaps', 'defense_snaps', 'st_snaps']
        if 'offense_pct' in self.snaps.columns and 'defense_pct' in self.snaps.columns:
            snap_cols.extend(['offense_pct', 'defense_pct'])

        available_snap_cols = [c for c in snap_cols if c in self.snaps.columns]
        snaps = self.snaps[available_snap_cols].copy()

        snaps['total_snaps'] = snaps['offense_snaps'].fillna(0) + snaps['defense_snaps'].fillna(0) + snaps['st_snaps'].fillna(0)

        # Compute snap percentage for the new metric
        if 'offense_pct' in snaps.columns and 'defense_pct' in snaps.columns:
            snaps['snap_pct'] = snaps[['offense_pct', 'defense_pct']].max(axis=1).fillna(0)
        else:
            # Fallback estimation if standard percentage columns are missing
            snaps['snap_pct'] = ((snaps['offense_snaps'].fillna(0) + snaps['defense_snaps'].fillna(0)) / 70.0).clip(upper=1.0)

        # 3. Merge rosters and snaps
        roster_cols = ['gsis_id', 'pfr_id', 'season', 'week', 'position', 'age', 'weight', 'full_name', 'team']
        available_roster_cols = [col for col in roster_cols if col in self.rosters.columns]

        df = pd.merge(
            self.rosters[available_roster_cols], 
            snaps, left_on=['pfr_id', 'season', 'week'], right_on=['pfr_player_id', 'season', 'week'], how='left'
        )

        # Rename for cleaner output
        df.rename(columns={'full_name': 'player_name'}, inplace=True)

        # 4. Process injuries (Target Creation)
        inj = self.injuries[['gsis_id', 'season', 'week', 'practice_status', 'report_status']].copy()

        # FIX 1: Deduplicate to prevent merge explosion
        inj = inj.drop_duplicates(subset=['gsis_id', 'season', 'week'], keep='last')

        # FIX 2: Define actual missed games (e.g., officially declared 'Out', 'Doubtful', or 'IR')
        # Note: You can refine this logic to check against total_snaps == 0 later in the pipeline
        inj['was_injured'] = inj['report_status'].isin(['Out', 'Reserve/Injured', 'Doubtful']).astype(int)

        inj['is_LP'] = (inj['practice_status'] == 'Limited Participation in Practice').astype(int)
        inj['is_DNP'] = (inj['practice_status'] == 'Did Not Participate In Practice').astype(int)

        # Merge injury info into main df
        df = pd.merge(df, inj, on=['gsis_id', 'season', 'week'], how='left')
        df['was_injured'] = df['was_injured'].fillna(0)
        df['is_LP'] = df['is_LP'].fillna(0)
        df['is_DNP'] = df['is_DNP'].fillna(0)

        # Sort for rolling calculations
        df = df.sort_values(['gsis_id', 'season', 'week'])

        # Compute Rolling Features (These can safely bridge the offseason boundary to capture momentum)
        df['total_snaps'] = df['total_snaps'].fillna(0)
        df['snaps_4wk_avg'] = df.groupby('gsis_id')['total_snaps'].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        ).fillna(0)

        df['snap_pct'] = df['snap_pct'].fillna(0)
        df['snap_pct_4wk_avg'] = df.groupby('gsis_id')['snap_pct'].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        ).fillna(0)

        df['career_injuries'] = df.groupby('gsis_id')['was_injured'].transform(
            lambda x: x.shift(1).cumsum()
        ).fillna(0)

        # FIX: Group targets by gsis_id AND season so Week 18 doesn't leak into Week 1 of the following year
        df['target_injured_next_week'] = df.groupby(['gsis_id', 'season'])['was_injured'].shift(-1)
        df['target_snap_pct_next_week'] = df.groupby(['gsis_id', 'season'])['snap_pct'].shift(-1)

        # Fill missing biometrics (impute with median for modeling)
        df['age'] = df['age'].fillna(df['age'].median() if not pd.isna(df['age'].median()) else 26.0)
        df['weight'] = df['weight'].fillna(df['weight'].median() if not pd.isna(df['weight'].median()) else 215.0)

        self.df = df

    def _train_next_week_model(self):
        """
        Trains calibrated XGBoost to output next week injury probability.
        """
        print("Training Next-Week Injury Probability Model (Task 1)...")
        train_df = self.df.dropna(subset=['target_injured_next_week']).reset_index(drop=True)

        X = train_df[self.features]
        y = train_df['target_injured_next_week']
        groups = train_df['season']

        base_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        # Explicitly generate train/test index splits using GroupKFold
        gkf = GroupKFold(n_splits=3)
        cv_splits = list(gkf.split(X, y, groups=groups))

        self.calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=cv_splits)
        self.calibrated_xgb.fit(X, y)

        # Generate predictions for the full dataset (as a percentage 0-100)
        self.df['in_game_injury_probability_pct'] = self.calibrated_xgb.predict_proba(self.df[self.features])[:, 1] * 100

    def _train_snap_pct_model(self):
        """
        Trains an XGBRegressor to estimate the percentage of snaps a player will play if active.
        """
        print("Training Projected Snap Percentage Model (Task 3)...")

        # Filter to instances where we know the next week's outcome
        train_df = self.df.dropna(subset=['target_snap_pct_next_week', 'target_injured_next_week']).reset_index(drop=True)
        
        # Only train on players who are ACTIVE next week to learn "snap pct IF active"
        train_df = train_df[train_df['target_injured_next_week'] == 0]

        # Include the rolling snap pct as a feature for this specific model
        model_features = self.features + ['snap_pct_4wk_avg']
        X = train_df[model_features]
        y = train_df['target_snap_pct_next_week']

        self.snap_regressor = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        self.snap_regressor.fit(X, y)

        # Predict on full dataset and convert to percentage 0-100
        self.df['projected_snap_pct_if_active'] = self.snap_regressor.predict(self.df[model_features]).clip(0, 1) * 100

    def _train_season_games_missed_model(self):
        """
        Aggregates to player-season and uses Zero-Inflated Negative Binomial
        to estimate total expected games missed.
        """
        print("Training Expected Season Games Missed Model (Task 2)...")

        # Aggregate to Player-Season level
        season_df = self.df.groupby(['gsis_id', 'season']).agg(
            total_games_missed=('was_injured', 'sum'),
            career_injuries=('career_injuries', 'max'),
            age=('age', 'max'),
            avg_weekly_snaps=('total_snaps', 'mean')
        ).reset_index()

        # FIX: Exclude the current season from training if we are mid-season
        # (Meaning the season is incomplete and 'total_games_missed' is artificially low)
        latest_season = self.df['season'].max()
        if self.end_week < 17:
            train_season_df = season_df[season_df['season'] < latest_season].copy()
        else:
            train_season_df = season_df.copy()

        formula = "total_games_missed ~ age + career_injuries + avg_weekly_snaps"

        # Add sm.add_constant to prevent missing intercept in inflation component
        exog_infl_train = sm.add_constant(train_season_df[['age', 'career_injuries']], has_constant='add')

        # Use ZeroInflatedNegativeBinomialP via .from_formula
        model = ZeroInflatedNegativeBinomialP.from_formula(
            formula, 
            data=train_season_df, 
            exog_infl=exog_infl_train
        )
        self.zinb_model = model.fit(disp=False)

        # Predict for ALL seasons (including the current partial season) to map back to the main DataFrame
        exog_infl_all = sm.add_constant(season_df[['age', 'career_injuries']], has_constant='add')
        season_df['projected_games_missed'] = self.zinb_model.predict(
            exog=season_df, exog_infl=exog_infl_all
        ).clip(lower=0, upper=17)

        # Map back to main DataFrame
        self.df = pd.merge(
            self.df, 
            season_df[['gsis_id', 'season', 'projected_games_missed']],
            on=['gsis_id', 'season'], 
            how='left'
        )

    def run_pipeline(self, output_path=OUTPUT_PATH):
        self._load_nflverse_data()
        self._engineer_features()
        self._train_next_week_model()
        self._train_season_games_missed_model()
        self._train_snap_pct_model()

        latest_season = self.end_season
        target_week = self.end_week

        # DYNAMIC CHECK:
        # If the run targets the final week of the regular season (e.g., Week 18),
        # dynamically transition to project Week 1 of the following season.
        if target_week >= 17:
            print(f"\nDetected end-of-season context (Season {latest_season}, Week {target_week}).")
            print("Applying offseason rollover to project Week 1 of the upcoming season...")

            export_df = self.df[
                (self.df['season'] == latest_season) &
                (self.df['week'] == target_week)
            ].copy()

            # Advance season and set to Week 1
            export_df['season'] = latest_season + 1
            export_df['week'] = 1

            # Increment age and clear stale end-of-year practice injury flags
            export_df['age'] = export_df['age'] + 1.0
            export_df['is_LP'] = 0
            export_df['is_DNP'] = 0

            # Re-evaluate predictions with the clean offseason parameters
            model_features = self.features
            snap_features = self.features + ['snap_pct_4wk_avg']

            export_df['in_game_injury_probability_pct'] = (
                self.calibrated_xgb.predict_proba(export_df[model_features])[:, 1] * 100
            )
            export_df['projected_snap_pct_if_active'] = (
                self.snap_regressor.predict(export_df[snap_features]).clip(0, 1) * 100
            )

            # --- FIX: Recalculate season games missed for the new year ---
            # 1. Grab their average weekly snaps from the season that just ended to feed the model
            avg_snaps = self.df[self.df['season'] == latest_season].groupby('gsis_id')['total_snaps'].mean().reset_index()
            avg_snaps.rename(columns={'total_snaps': 'avg_weekly_snaps'}, inplace=True)
            export_df = export_df.merge(avg_snaps, on='gsis_id', how='left').fillna({'avg_weekly_snaps': 0})

            # 2. Rebuild the inflation features with the required constant column
            exog_infl_new = sm.add_constant(export_df[['age', 'career_injuries']], has_constant='add')

            # 3. Predict new games missed for the upcoming season
            export_df['projected_games_missed'] = self.zinb_model.predict(
                exog=export_df, exog_infl=exog_infl_new
            ).clip(lower=0, upper=17)

            proj_season = latest_season + 1
            proj_week = 1
        else:
            # Normal in-season behavior: predict the next immediate week using actual current status flags
            export_df = self.df[
                (self.df['season'] == latest_season) &
                (self.df['week'] == target_week)
            ].copy()
            proj_season = latest_season
            proj_week = target_week + 1
            print(f"\nIn-season context detected. Exporting projections for Season {proj_season}, Week {proj_week}.")

        # 1. Round specified numerical columns to 2 decimal places
        round_cols = [
            'in_game_injury_probability_pct',
            'projected_games_missed',
            'projected_snap_pct_if_active'
        ]
        export_df[round_cols] = export_df[round_cols].round(2)

        # 2. Determine 'injury_risk' based on the distribution of 'projected_games_missed'
        export_df['injury_risk'] = pd.qcut(
            export_df['projected_games_missed'],
            q=4,
            labels=['Minimal', 'Low', 'Moderate', 'Elevated'],
            duplicates='drop'
        )

        out_cols = [
            'player_name', 'team', 'position', 'season', 'week', 
            'in_game_injury_probability_pct', 'projected_games_missed',
            'projected_snap_pct_if_active', 'injury_risk', 'gsis_id'
        ]

        # Create output directory if it doesn't exist to prevent file write errors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        export_df['season'] = proj_season
        export_df['week'] = proj_week

        export_df[out_cols].to_csv(output_path, index=False)
        print(f"Pipeline complete! Saved predictions to {output_path}")


if __name__ == "__main__":
    predictor = NFLInjuryPredictor()
    predictor.run_pipeline()
