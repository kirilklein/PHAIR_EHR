import logging
import time
from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    COMBINATIONS,
    CONCEPT_COL,
    PID_COL,
    PRIMARY,
    SECONDARY,
    TIMESTAMP_COL,
    TIMESTAMP_SOURCE,
    VALUE_COL,
    WINDOW_HOURS_MAX,
    WINDOW_HOURS_MIN,
)
from corebehrt.functional.cohort_handling.combined_outcomes import (
    check_combination_args,
    create_empty_results_df,
    find_matches_within_window,
)
from corebehrt.functional.cohort_handling.matching import get_col_booleans
from corebehrt.functional.preparation.filter import remove_missing_timestamps
from corebehrt.functional.utils.time import get_hours_since_epoch

logger = logging.getLogger(__name__)


class OutcomeMaker:
    def __init__(self, outcomes: dict):
        self.outcomes = outcomes
        logger.info(f"Number of outcomes: {len(outcomes)}")
        self.write_header = {
            outcome: True for outcome in outcomes
        }  # write header for all outcomes on first call

    def __call__(
        self,
        concepts_plus: pd.DataFrame,
        outcomes_path: str,
    ) -> None:
        """Create outcomes from concepts_plus and patients_info and writes them to disk."""
        concepts_plus = self._prepare_concepts_plus(concepts_plus)
        total_time_start = time.time()
        for outcome, attrs in self.outcomes.items():
            start_time = time.time()
            # Handle combination outcomes
            if COMBINATIONS in attrs:
                check_combination_args(attrs[COMBINATIONS])
                timestamps = self.match_combinations(concepts_plus, attrs[COMBINATIONS])
            # Handle traditional outcomes
            else:
                types = attrs["type"]
                matches = attrs["match"]
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)

            if "exclusion" in attrs and not timestamps.empty:
                timestamps = self._apply_exclusion_filter(
                    concepts_plus, timestamps, attrs["exclusion"]
                )

            self._write_df(timestamps, outcomes_path, outcome)
            end_time = time.time()
            logger.info(f"Time taken for {outcome}: {end_time - start_time} seconds")
        total_time_end = time.time()
        logger.info(
            f"Total time taken for batch: {total_time_end - total_time_start} seconds"
        )

    @staticmethod
    def _prepare_concepts_plus(concepts_plus: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare concepts_plus for matching.
        - Remove missing timestamps
        - Get only relevant columns
        - Add absolute positions
        - Convert absolute positions to int
        - Convert concept column to categorical
        """
        concepts_plus = remove_missing_timestamps(concepts_plus)
        concepts_plus = concepts_plus[
            [PID_COL, TIMESTAMP_COL, CONCEPT_COL, VALUE_COL]
        ]  # get only relevant columns
        concepts_plus[ABSPOS_COL] = get_hours_since_epoch(concepts_plus[TIMESTAMP_COL])
        concepts_plus[ABSPOS_COL] = concepts_plus[ABSPOS_COL].astype(int)
        # concepts_plus[CONCEPT_COL] = pd.Categorical(concepts_plus[CONCEPT_COL])
        return concepts_plus

    def _write_df(
        self,
        timestamps: pd.DataFrame,
        outcomes_path: str,
        outcome: str,
    ):
        """
        Write a dataframe to a csv file. If the file does not exist, create it and write the header.
        If the file exists, append the data.
        If the dataframe is empty, do nothing.
        """
        output_path = join(outcomes_path, f"{outcome}.csv")
        if timestamps.empty:
            if self.write_header[outcome]:
                logger.warning(f"Outcome {outcome} has no data. Write only header.")
                pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL]).to_csv(
                    output_path, header=True, index=False
                )
                self.write_header[outcome] = False
            return  # always return, but write only on first call

        write_header = self.write_header[outcome]
        mode = "w" if write_header else "a"
        timestamps.to_csv(output_path, mode=mode, header=write_header, index=False)
        self.write_header[outcome] = False  # ! important for next iterations

    def match_concepts(
        self,
        concepts_plus: pd.DataFrame,
        types: List[List],
        matches: List[List],
        attrs: Dict,
    ) -> pd.DataFrame:
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE
        """
        # Handle empty DataFrame
        if len(concepts_plus) == 0:
            return create_empty_results_df()

        filtered_concepts = concepts_plus

        if "exclude" in attrs:
            filtered_concepts = filtered_concepts[
                ~filtered_concepts[CONCEPT_COL].isin(attrs["exclude"])
            ]

        col_booleans = get_col_booleans(
            filtered_concepts,
            types,
            matches,
            attrs.get("match_how", "startswith"),
            attrs.get("case_sensitive", True),
        )

        if len(col_booleans) == 0:
            return create_empty_results_df()

        mask = np.bitwise_and.reduce(col_booleans)

        if "negation" in attrs:
            mask = ~mask

        result = filtered_concepts[mask]
        if len(result) > 0:
            return result.drop(columns=[CONCEPT_COL, VALUE_COL])
        else:
            return create_empty_results_df()

    def match_combinations(
        self,
        concepts_plus: pd.DataFrame,
        combinations: Dict,
    ) -> pd.DataFrame:
        """Match combinations of codes that occur within a specific time window of each other.

        Args:
            concepts_plus: DataFrame containing concepts
            combinations: Dictionary defining the combinations to match
                Example: {
                    "primary": {"type": ["code"], "match": [["DOD"]]},
                    "secondary": {"type": ["code"], "match": [["DI20"]]},
                    "window_hours_min": 24,
                    "window_hours_max": 24,
                    "timestamp_source": "primary" # or "secondary"
                }

        Returns:
            DataFrame with timestamps of matched combinations
        """
        # Handle empty input DataFrame
        if len(concepts_plus) == 0:
            return create_empty_results_df()

        # Get primary and secondary events
        primary_events = self.get_events(concepts_plus, combinations[PRIMARY])
        secondary_events = self.get_events(concepts_plus, combinations[SECONDARY])

        # Return empty DataFrame if either set of events is empty
        if len(primary_events) == 0 or len(secondary_events) == 0:
            return create_empty_results_df()

        # Find events within the time window
        return find_matches_within_window(
            primary_events,
            secondary_events,
            window_hours_min=combinations[WINDOW_HOURS_MIN],
            window_hours_max=combinations[WINDOW_HOURS_MAX],
            timestamp_source=combinations.get(TIMESTAMP_SOURCE, PRIMARY),
        )

    def get_events(self, concepts_plus: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Extract events from concepts based on configuration."""
        extra_params = {k: v for k, v in config.items() if k not in ["type", "match"]}
        return self.match_concepts(
            concepts_plus, config["type"], config["match"], extra_params
        )

    def _apply_exclusion_filter(
        self,
        concepts_plus: pd.DataFrame,
        target_events: pd.DataFrame,
        exclusion_config: Dict,
    ) -> pd.DataFrame:
        """Filter out target events that co-occur with exclusion events within a specified time window."""
        exclusion_events = self.get_events(concepts_plus, exclusion_config["events"])

        if exclusion_events.empty or target_events.empty:
            return target_events

        target_events = target_events.copy()
        exclusion_events = exclusion_events.rename(
            columns={ABSPOS_COL: f"{ABSPOS_COL}_exclusion"}
        )

        target_events_with_index = target_events.reset_index()
        merged = pd.merge(
            target_events_with_index,
            exclusion_events[[PID_COL, f"{ABSPOS_COL}_exclusion"]],
            on=PID_COL,
        )

        time_diff = merged[f"{ABSPOS_COL}_exclusion"] - merged[ABSPOS_COL]
        window_min = exclusion_config["window_hours_min"]
        window_max = exclusion_config["window_hours_max"]

        mask = (time_diff >= window_min) & (time_diff <= window_max)
        indices_to_drop = merged.loc[mask, "index"].unique()

        return target_events.drop(index=indices_to_drop)
