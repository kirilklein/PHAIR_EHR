from typing import Dict, List, Tuple

import pandas as pd

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CODE_ENTRY,
    CODE_GROUPS,
    CODE_MASK,
    CRITERION_FLAG,
    DAYS,
    EXCLUDE_CODES,
    EXPRESSION,
    FINAL_MASK,
    INDEX_DATE,
    MAX_AGE,
    MAX_VALUE,
    MIN_AGE,
    MIN_VALUE,
    NUMERIC_VALUE,
    TIME_MASK,
    TIME_WINDOW_DAYS,
)
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.cohort_handling.advanced.checks import (
    check_criteria_definitions,
    check_delays_config,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    compute_age_at_index_date,
    compute_code_masks,
    compute_delay_column,
    compute_time_mask,
    compute_time_window_columns,
    extract_criteria_names_from_expression,
    extract_numeric_values,
    merge_index_dates,
    rename_result,
)


class CohortExtractor:
    def __init__(self, criteria_definitions: dict, delays_config: dict = None):
        self.criteria_definitions = criteria_definitions
        self.delays_config = delays_config or {}

        check_delays_config(self.delays_config)
        check_criteria_definitions(self.criteria_definitions)

    def extract(
        self,
        events: pd.DataFrame,
        index_dates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract and evaluate criteria for each patient in a vectorized fashion.

        This function processes "simple" criteria (code- and numeric-based) first, then computes
        additional criteria:
        - Composite criteria defined via an expression (EXPRESSION key)
        - Age-based criteria (using MIN_AGE and/or MAX_AGE)

        Also computes the patient's age at index_date and merges it into the final results.

        Returns a DataFrame with one row per patient (subject_id) and one column per criterion.
        Additionally, an "age_at_index_date" column is added.
        """
        relevant_index_dates = self._get_relevant_index_dates(events, index_dates)
        print(f"Processing {len(relevant_index_dates)} patients")

        age_df = compute_age_at_index_date(relevant_index_dates, events)

        simple_criteria, additional_criteria = self._partition_criteria()
        simple_results = self._process_simple_criteria(
            events, relevant_index_dates, simple_criteria
        )

        all_results = simple_results.merge(age_df, on=PID_COL, how="left")
        all_results = self._process_additional_criteria(
            all_results, additional_criteria
        )

        return all_results

    def _get_relevant_index_dates(
        self, events: pd.DataFrame, index_dates: pd.DataFrame
    ) -> pd.DataFrame:
        shard_ids = events[PID_COL].unique()
        relevant_dates = index_dates[index_dates[PID_COL].isin(shard_ids)]
        return relevant_dates.rename(columns={TIMESTAMP_COL: INDEX_DATE})

    def _partition_criteria(self) -> Tuple[List, List]:
        simple_criteria, additional_criteria = [], []

        for criterion, crit_cfg in self.criteria_definitions.items():
            if CODE_ENTRY in crit_cfg:
                simple_criteria.append((criterion, crit_cfg))
            elif any(k in crit_cfg for k in [MIN_AGE, MAX_AGE, EXPRESSION]):
                additional_criteria.append((criterion, crit_cfg))

        return simple_criteria, additional_criteria

    def _process_simple_criteria(
        self,
        events: pd.DataFrame,
        relevant_index_dates: pd.DataFrame,
        simple_criteria: list,
    ) -> pd.DataFrame:
        results = []

        for criterion, crit_cfg in simple_criteria:
            res = self._vectorized_extraction_codes(
                events, relevant_index_dates, crit_cfg
            )
            has_numeric = NUMERIC_VALUE in crit_cfg
            res = rename_result(res, criterion, has_numeric)
            results.append(res)

        if not results:
            return pd.DataFrame({PID_COL: relevant_index_dates[PID_COL].unique()})

        combined = results[0]
        for df in results[1:]:
            combined = combined.merge(df, on=PID_COL, how="outer")

        return combined

    def _process_additional_criteria(
        self, all_results: pd.DataFrame, additional_criteria: list
    ) -> pd.DataFrame:
        """Process composite and age-based criteria and merge with existing results."""

        results = []

        for criterion, crit_cfg in additional_criteria:
            if EXPRESSION in crit_cfg:
                res = self._vectorized_extraction_expression(
                    crit_cfg[EXPRESSION], all_results
                )
            elif MIN_AGE in crit_cfg or MAX_AGE in crit_cfg:
                res = self._vectorized_extraction_age(
                    all_results, crit_cfg.get(MIN_AGE), crit_cfg.get(MAX_AGE)
                )
            else:
                continue

            res = res.rename(columns={CRITERION_FLAG: criterion})[[PID_COL, criterion]]
            results.append(res)

        if not results:
            return all_results

        combined = results[0]
        for df in results[1:]:
            combined = combined.merge(df, on=PID_COL, how="outer")

        return all_results.merge(combined, on=PID_COL, how="left")

    def _vectorized_extraction_codes(
        self, events: pd.DataFrame, index_dates: pd.DataFrame, crit_cfg: Dict
    ) -> pd.DataFrame:
        """
        Fully vectorized extraction for a code/numeric criterion.

        Steps:
        1. Merge index_dates into events.
        2. Compute per-event delay using delays_config.
        3. Compute the time window (min_timestamp and max_timestamp).
        4. Build time and code masks and combine them into FINAL_MASK.
        5. Group by patient: a patient's CRITERION_FLAG is True if any event passes FINAL_MASK.
        6. If numeric extraction is requested, additionally extract, for each patient, the latest numeric_value
            (subject to additional range filtering on min_value and/or max_value).

        Returns a DataFrame with columns: PID_COL, CRITERION_FLAG, and (if applicable) NUMERIC_VALUE.
        """
        df = merge_index_dates(events, index_dates)
        df = compute_delay_column(
            df, self.delays_config.get(CODE_GROUPS, []), self.delays_config.get(DAYS, 0)
        )
        df = compute_time_window_columns(df, crit_cfg.get(TIME_WINDOW_DAYS, 36500))

        df[TIME_MASK] = compute_time_mask(df)
        df[CODE_MASK] = compute_code_masks(
            df, crit_cfg[CODE_ENTRY], crit_cfg.get(EXCLUDE_CODES, [])
        )
        df[FINAL_MASK] = df[TIME_MASK] & df[CODE_MASK]

        flag_df = (
            df.groupby(PID_COL)[FINAL_MASK]
            .any()
            .reset_index()
            .rename(columns={FINAL_MASK: CRITERION_FLAG})
        )

        if NUMERIC_VALUE in crit_cfg:
            min_val = crit_cfg.get(MIN_VALUE)
            max_val = crit_cfg.get(MAX_VALUE)
            result = extract_numeric_values(df, flag_df, min_val, max_val)
        else:
            result = flag_df.copy()
            result[NUMERIC_VALUE] = None

        return result

    @staticmethod
    def _vectorized_extraction_expression(
        expression: str, initial_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate a composite criterion defined by an expression.

        Args:
            expression: String expression using boolean operators (&, |, ~, and, or, not)
            initial_results: DataFrame containing the criteria columns to evaluate

        Returns:
            DataFrame with columns: PID_COL and CRITERION_FLAG
        """
        # First, replace any '~' with 'not ' to ensure proper spacing
        expression = expression.replace("~", "not ")

        # Extract criteria names (this will now exclude the ~ operator)
        expression_criteria = extract_criteria_names_from_expression(expression)

        # Construct the local dict for evaluation
        local_dict = CohortExtractor._construct_local_dict(
            initial_results, expression_criteria
        )

        # Evaluate the expression
        composite_flag = pd.eval(expression, local_dict=local_dict)

        result = initial_results[[PID_COL]].copy()
        result[CRITERION_FLAG] = composite_flag
        return result

    @staticmethod
    def _construct_local_dict(
        initial_results: pd.DataFrame, expression_criteria: list
    ) -> dict:
        """Construct a local dictionary for the expression evaluation."""
        local_dict = {}
        for criterion in expression_criteria:
            series = initial_results[criterion]
            # Check if the column is boolean or numeric using pandas type utilities.
            if not (
                pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_numeric_dtype(series)
            ):
                raise ValueError(
                    f"Column '{criterion}' must be boolean or numeric for expression evaluation."
                )
            # Convert to boolean.
            local_dict[criterion] = series.astype(bool)
        return local_dict

    def _vectorized_extraction_age(
        self, initial_results: pd.DataFrame, min_age: int = None, max_age: int = None
    ) -> pd.DataFrame:
        """
        Evaluate an age-based criterion.

        Returns a DataFrame with columns:
        PID_COL and CRITERION_FLAG.
        """
        df = initial_results[[PID_COL, AGE_AT_INDEX_DATE]].copy()
        flag = pd.Series(True, index=df.index)
        if min_age is not None:
            flag &= df[AGE_AT_INDEX_DATE] >= min_age
        if max_age is not None:
            flag &= df[AGE_AT_INDEX_DATE] <= max_age

        result = df[[PID_COL]].copy()
        result[CRITERION_FLAG] = flag
        return result
