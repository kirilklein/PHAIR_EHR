import logging
import time
import json
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

        # Initialize stats tracking
        self.stats = {
            outcome: {"total_outcomes": 0, "unique_subjects": set()}
            for outcome in outcomes
        }

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
            timestamps = create_empty_results_df()  # Default empty df
            # Handle combination outcomes
            if COMBINATIONS in attrs:
                combinations_config = attrs[COMBINATIONS]
                check_combination_args(combinations_config)
                is_exclusion = combinations_config.get("exclude", False)
                timestamps = self.match_combinations(
                    concepts_plus, combinations_config, exclude=is_exclusion
                )
            else:
                timestamps = self.match_concepts(
                    concepts_plus, attrs["type"], attrs["match"], attrs
                )

            # Update stats before writing
            self._update_stats(outcome, timestamps)

            self._write_df(timestamps, outcomes_path, outcome)
            end_time = time.time()
            logger.info(f"Time taken for {outcome}: {end_time - start_time} seconds")
        total_time_end = time.time()
        logger.info(
            f"Total time taken for batch: {total_time_end - total_time_start} seconds"
        )

        # Log cumulative stats and save to JSON
        self._log_stats()
        self._save_stats_to_json(outcomes_path)

    @staticmethod
    def _prepare_concepts_plus(concepts_plus: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare concepts_plus for matching.
        - Remove missing timestamps
        - Get only relevant columns
        - Add absolute positions
        - Convert concept column to categorical
        - Convert value column to string for consistent matching
        """
        concepts_plus = remove_missing_timestamps(concepts_plus)
        concepts_plus = concepts_plus[
            [PID_COL, TIMESTAMP_COL, CONCEPT_COL, VALUE_COL]
        ]  # get only relevant columns
        concepts_plus[ABSPOS_COL] = get_hours_since_epoch(concepts_plus[TIMESTAMP_COL])
        concepts_plus[CONCEPT_COL] = pd.Categorical(concepts_plus[CONCEPT_COL])
        # Pre-compute a lowercase version for fast case-insensitive matching
        concepts_plus[f"{CONCEPT_COL}_lower"] = (
            concepts_plus[CONCEPT_COL].astype(str).str.lower()
        )
        # Convert VALUE_COL to string for consistent matching
        concepts_plus[VALUE_COL] = concepts_plus[VALUE_COL].astype(str)
        # Pre-compute a lowercase version for case-insensitive value matching
        concepts_plus[f"{VALUE_COL}_lower"] = concepts_plus[VALUE_COL].str.lower()
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
        write_columns = [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
        if timestamps.empty:
            if self.write_header[outcome]:
                logger.warning(f"Outcome {outcome} has no data. Write only header.")
                pd.DataFrame(columns=write_columns).to_csv(
                    output_path, header=True, index=False
                )
                self.write_header[outcome] = False
            return  # always return, but write only on first call

        write_header = self.write_header[outcome]
        mode = "w" if write_header else "a"
        timestamps[write_columns].to_csv(
            output_path, mode=mode, header=write_header, index=False
        )
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

    def _find_matches_fast(
        self,
        primary_events: pd.DataFrame,
        secondary_events: pd.DataFrame,
        window_hours_min: int,
        window_hours_max: int,
        timestamp_source: str = PRIMARY,
    ) -> pd.DataFrame:
        """The single, fast engine for finding all event pairs within a time window."""
        if primary_events.empty or secondary_events.empty:
            return create_empty_results_df()

        primary_with_index = primary_events.reset_index()
        secondary_renamed = secondary_events.rename(
            columns={
                TIMESTAMP_COL: f"{TIMESTAMP_COL}_{SECONDARY}",
                ABSPOS_COL: f"{ABSPOS_COL}_{SECONDARY}",
            }
        )
        merged = pd.merge(
            primary_with_index,
            secondary_renamed[
                [PID_COL, f"{TIMESTAMP_COL}_{SECONDARY}", f"{ABSPOS_COL}_{SECONDARY}"]
            ],
            on=PID_COL,
        )
        time_diff = merged[f"{ABSPOS_COL}_{SECONDARY}"] - merged[ABSPOS_COL]
        mask = (time_diff >= window_hours_min) & (time_diff <= window_hours_max)
        valid_matches = merged[mask]

        if valid_matches.empty:
            return create_empty_results_df()

        valid_matches = valid_matches.drop_duplicates(subset="index")

        if timestamp_source == SECONDARY:
            valid_matches[TIMESTAMP_COL] = valid_matches[f"{TIMESTAMP_COL}_{SECONDARY}"]
            valid_matches[ABSPOS_COL] = valid_matches[f"{ABSPOS_COL}_{SECONDARY}"]

        result = valid_matches.set_index("index")[primary_events.columns]
        return result

    def match_combinations(
        self,
        concepts_plus: pd.DataFrame,
        combinations: Dict,
        exclude: bool = False,
    ) -> pd.DataFrame:
        """
        Finds combinations or exclusions based on the exclude flag.

        If exclude is False, it KEEPS the primary events that have a match.
        If exclude is True, it DROPS the primary events that have a match.
        """
        primary_events = self.get_events(concepts_plus, combinations[PRIMARY])
        if primary_events.empty:
            return create_empty_results_df()

        secondary_events = self.get_events(concepts_plus, combinations[SECONDARY])

        if secondary_events.empty:
            # If we are excluding, no secondary events means nothing to drop, so return all primary events.
            # If we are combining, no secondary events means no matches, so return empty.
            return primary_events if exclude else create_empty_results_df()

        # Find all primary events that have a valid secondary match
        matching_events = self._find_matches_fast(
            primary_events,
            secondary_events,
            window_hours_min=combinations[WINDOW_HOURS_MIN],
            window_hours_max=combinations[WINDOW_HOURS_MAX],
            timestamp_source=combinations.get(TIMESTAMP_SOURCE, PRIMARY),
        )
        if exclude:
            # For exclusion, drop the matching events from the original primary set
            if not matching_events.empty:
                return primary_events.drop(index=matching_events.index)
            else:
                return primary_events  # No matches found, so nothing to exclude
        else:
            # For combination, simply return the events that were matched
            return matching_events

    def get_events(self, concepts_plus: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Extract events from concepts based on configuration."""
        extra_params = {k: v for k, v in config.items() if k not in ["type", "match"]}
        return self.match_concepts(
            concepts_plus, config["type"], config["match"], extra_params
        )

    def _update_stats(self, outcome: str, timestamps: pd.DataFrame) -> None:
        """Update statistics for an outcome."""
        if not timestamps.empty:
            # Update total count
            batch_count = len(timestamps)
            self.stats[outcome]["total_outcomes"] += batch_count

            # Update unique subjects
            unique_subjects_in_batch = set(timestamps[PID_COL].unique())
            self.stats[outcome]["unique_subjects"].update(unique_subjects_in_batch)

            logger.info(
                f"Outcome {outcome}: +{batch_count} outcomes, "
                f"+{len(unique_subjects_in_batch)} subjects in this batch"
            )

    def _log_stats(self) -> None:
        """Log cumulative statistics for all outcomes."""
        logger.info("=== Cumulative Outcome Statistics ===")
        for outcome, stats in self.stats.items():
            total_outcomes = stats["total_outcomes"]
            unique_subject_count = len(stats["unique_subjects"])
            logger.info(
                f"{outcome}: {total_outcomes} total outcomes, "
                f"{unique_subject_count} unique subjects"
            )

    def _save_stats_to_json(self, outcomes_path: str) -> None:
        """Save cumulative statistics to a JSON file."""
        stats_for_json = self.get_stats()
        stats_file_path = join(outcomes_path, "outcome_statistics.json")

        try:
            with open(stats_file_path, "w") as f:
                json.dump(stats_for_json, f, indent=2)
            logger.info(f"Outcome statistics saved to: {stats_file_path}")
        except Exception as e:
            logger.error(f"Failed to save outcome statistics to JSON: {e}")

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Return current statistics as a dictionary."""
        return {
            outcome: {
                "total_outcomes": stats["total_outcomes"],
                "unique_subjects": len(stats["unique_subjects"]),
            }
            for outcome, stats in self.stats.items()
        }
