"""
Module for extracting and evaluating complex patient criteria from medical event data.

This module provides classes and functions to:
1. Extract patient criteria based on medical codes, numeric values, and time windows
2. Evaluate age-based criteria relative to index dates
3. Process hierarchical criteria expressions with dependencies
4. Handle complex temporal relationships through delay configurations

Key classes:
- CohortExtractor: Main class for orchestrating criteria extraction
- ExpressionCriteriaResolver: Handles hierarchical criteria dependencies
- CriteriaExtraction: Static methods for different types of criteria evaluation

Example usage:
```python
extractor = CohortExtractor(
    criteria_definitions={
        "diabetes": {
            "code_entry": ["E11%"],
            "start_days": -365,  # 365 days before index date
            "end_days": 0        # Up to (not including) index date
        },
        "post_index_meds": {
            "code_entry": ["R%"],
            "start_days": 0,     # From index date
            "end_days": 30       # Up to 30 days after index date
        },
        "elderly": {
            "min_age": 65
        },
        "high_risk": {
            "expression": "diabetes & elderly"
        }
    }
)

results = extractor.extract(events_df, index_dates_df)
```

The module supports three types of criteria:
1. Code-based: Match medical codes with optional time windows and numeric thresholds
2. Age-based: Filter by age at index date
3. Expression-based: Combine other criteria using boolean expressions
"""

import logging
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CODE_ENTRY,
    CODE_MASK,
    CRITERION_FLAG,
    EXCLUDE_CODES,
    EXPRESSION,
    EXTRACT_VALUE,
    FINAL_MASK,
    INDEX_DATE,
    MAX_AGE,
    MAX_VALUE,
    MIN_AGE,
    MIN_TIME,
    MAX_TIME,
    MIN_VALUE,
    NUMERIC_VALUE,
    START_DAYS,
    END_DAYS,
    TIME_MASK,
    UNIQUE_CRITERIA_LIST,
    MIN_COUNT,
    MAX_COUNT,
)
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.cohort_handling.advanced.extract import (
    compute_age_at_index_date,
    compute_code_masks,
    extract_criteria_names_from_expression,
    extract_numeric_values,
    merge_index_dates,
    rename_result,
    compute_time_mask_exclusive,
)

logger = logging.getLogger("select_cohort_advanced")


class CohortExtractor:
    def __init__(self, criteria_definitions: dict):
        self.criteria_definitions = criteria_definitions

    def extract(
        self,
        events: pd.DataFrame,
        index_dates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract and evaluate criteria for each patient in a vectorized fashion.

        Clearly separates the extraction into three steps:
        1. Simple (code/numeric-based) criteria
        2. Age-based criteria
        3. Expression-based criteria (hierarchical)
        """
        relevant_index_dates = self._get_relevant_index_dates(events, index_dates)
        logger.info(f"\tNumber of patients in index dates: {len(relevant_index_dates)}")
        logger.info(f"\tComputing age at index date")
        age_df = compute_age_at_index_date(relevant_index_dates, events)

        logger.info("\tPartitioning criteria")
        simple_criteria, age_criteria, expression_criteria, count_criteria = (
            self._partition_criteria()
        )

        logger.info("\tProcessing simple criteria")
        simple_results = self._process_simple_criteria(
            events, relevant_index_dates, simple_criteria
        )

        all_results = simple_results.merge(
            age_df, on=PID_COL, how="inner", validate="one_to_one"
        )

        logger.info("\tProcessing age criteria")
        all_results = self._process_age_criteria(all_results, age_criteria)

        logger.info("\tProcessing count criteria")
        all_results = self._process_count_criteria(all_results, count_criteria)

        logger.info("\tProcessing expression criteria")
        all_results = self._process_expression_criteria(
            all_results, expression_criteria
        )
        logger.info("\tDone")
        return all_results

    def _get_relevant_index_dates(
        self, events: pd.DataFrame, index_dates: pd.DataFrame
    ) -> pd.DataFrame:
        shard_ids = events[PID_COL].unique()
        relevant_dates = index_dates[index_dates[PID_COL].isin(shard_ids)]
        return relevant_dates.rename(columns={TIMESTAMP_COL: INDEX_DATE})

    def _partition_criteria(self) -> Tuple[List, List, List]:
        """Separate criteria into simple, age-based, and expression-based criteria."""
        simple_criteria, age_criteria, expression_criteria, count_criteria = (
            [],
            [],
            [],
            [],
        )

        for criterion, crit_cfg in self.criteria_definitions.items():
            if CODE_ENTRY in crit_cfg:
                simple_criteria.append((criterion, crit_cfg))
            elif MIN_AGE in crit_cfg or MAX_AGE in crit_cfg:
                age_criteria.append((criterion, crit_cfg))
            elif EXPRESSION in crit_cfg:
                expression_criteria.append((criterion, crit_cfg))
            elif UNIQUE_CRITERIA_LIST in crit_cfg:
                count_criteria.append((criterion, crit_cfg))

        return simple_criteria, age_criteria, expression_criteria, count_criteria

    def _process_simple_criteria(
        self,
        events: pd.DataFrame,
        relevant_index_dates: pd.DataFrame,
        simple_criteria: list,
    ) -> pd.DataFrame:
        """Process multiple criteria against events data for given index dates.

        Args:
            events: DataFrame containing medical events with at least 'code' and PID columns
            relevant_index_dates: DataFrame with index dates for each patient ID
            simple_criteria: List of (criterion_name, criterion_config) tuples

        Returns:
            DataFrame with extracted criteria results per patient ID
        """
        if not simple_criteria:
            return pd.DataFrame({PID_COL: relevant_index_dates[PID_COL].unique()})

        combined_df = merge_index_dates(events, relevant_index_dates)

        results = []
        for criterion, crit_cfg in tqdm(
            simple_criteria,
            desc="Processing simple criteria",
        ):
            res = CriteriaExtraction.extract_codes(combined_df, crit_cfg)
            res = rename_result(res, criterion, NUMERIC_VALUE in crit_cfg)
            results.append(res)

        if results:
            return self.combine_results(results)
        return pd.DataFrame({PID_COL: relevant_index_dates[PID_COL].unique()})

    def _process_count_criteria(
        self,
        all_results: pd.DataFrame,
        count_criteria: list,
    ) -> pd.DataFrame:
        """
        Evaluate and merge count-based criteria into the cohort results.

        For each (name, crit_cfg) in count_criteria:
          - Confirms each UNIQUE_CRITERIA_LIST column exists in all_results.
          - Computes the rule’s boolean flag via extract_count_criteria.
          - Renames the flag column to the criterion name.
        Finally, left-joins all new flag columns back into all_results.

        Args:
            all_results: DataFrame with PID_COL and precomputed criteria flags.
            count_criteria: List of tuples (criterion_name, crit_cfg), where crit_cfg contains:
                UNIQUE_CRITERIA_LIST: List[str] of flag columns to count.
                MIN_COUNT (int, optional): Minimum required true flags (default 0).
                MAX_COUNT (int, optional): Maximum allowed true flags (default ∞).

        Returns:
            DataFrame: original all_results with added boolean columns for each count criterion.
        """
        if not count_criteria:
            return all_results

        results = []
        for criterion, crit_cfg in count_criteria:
            # Validate required flags exist
            missing = [
                c
                for c in crit_cfg.get(UNIQUE_CRITERIA_LIST, [])
                if c not in all_results.columns
            ]
            if missing:
                raise ValueError(
                    f"Count criterion '{criterion}' missing columns: {missing}."
                )

            # Extract and rename
            res = CriteriaExtraction.extract_count_criteria(all_results, crit_cfg)
            res = rename_result(res, criterion, False)
            results.append(res)

        # Merge all count criteria back into the main results
        combined = self.combine_results(results)
        return all_results.merge(
            combined, on=PID_COL, how="inner", validate="one_to_one"
        )

    @staticmethod
    def combine_results(results: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine results from different criteria."""
        all_pids = set()
        for df in results:
            all_pids.update(df[PID_COL])

        # Create a base DataFrame with all patient IDs
        combined = pd.DataFrame({PID_COL: list(all_pids)})

        # Left join each result DataFrame
        for df in results:
            combined = combined.merge(df, on=PID_COL, how="left", validate="one_to_one")

        return combined

    def _process_age_criteria(
        self, all_results: pd.DataFrame, age_criteria: list
    ) -> pd.DataFrame:
        """Process age-based criteria and merge with existing results."""
        results = []

        for criterion, crit_cfg in age_criteria:
            res = CriteriaExtraction.extract_age(
                all_results, crit_cfg.get(MIN_AGE), crit_cfg.get(MAX_AGE)
            )
            res = res.rename(columns={CRITERION_FLAG: criterion})[[PID_COL, criterion]]
            results.append(res)

        if not results:
            return all_results

        combined = results[0]
        for df in results[1:]:
            combined = combined.merge(df, on=PID_COL, how="outer")

        return all_results.merge(
            combined, on=PID_COL, how="inner", validate="one_to_one"
        )

    def _process_expression_criteria(
        self, all_results: pd.DataFrame, expression_criteria: list, max_iter: int = 5
    ) -> pd.DataFrame:
        """
        Process expression-based criteria hierarchically.
        """
        unresolved_criteria = expression_criteria.copy()
        resolved_criteria = set(all_results.columns)
        iteration = 0
        while iteration < max_iter:
            iteration += 1
            newly_resolved, unresolved_criteria = (
                ExpressionCriteriaResolver.resolve_iteration(
                    all_results, unresolved_criteria, resolved_criteria
                )
            )

            if not newly_resolved:
                break  # Stop if no further resolutions

            resolved_criteria.update(newly_resolved)

        ExpressionCriteriaResolver.ensure_all_resolved(unresolved_criteria, max_iter)

        return all_results

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


class ExpressionCriteriaResolver:
    @staticmethod
    def resolve_iteration(
        all_results: pd.DataFrame,
        criteria_to_resolve: list,
        resolved_criteria: set,
    ) -> Tuple[set, list]:
        """
        Attempt to resolve each criterion. Returns a set of resolved criterion names
        and a list of still unresolved criteria configurations.
        """
        newly_resolved = set()
        still_unresolved = []

        for criterion, crit_cfg in criteria_to_resolve:
            if ExpressionCriteriaResolver.can_resolve(crit_cfg, resolved_criteria):
                res = CriteriaExtraction.extract_expression(
                    crit_cfg[EXPRESSION], all_results
                )
                res = res.rename(columns={CRITERION_FLAG: criterion})[
                    [PID_COL, criterion]
                ]
                all_results[criterion] = res[criterion]
                newly_resolved.add(criterion)
            else:
                still_unresolved.append((criterion, crit_cfg))

        return newly_resolved, still_unresolved

    @staticmethod
    def can_resolve(crit_cfg: dict, resolved_criteria: set) -> bool:
        """
        Check if criterion expression dependencies are resolved.
        """
        if EXPRESSION in crit_cfg:
            expr_criteria = extract_criteria_names_from_expression(crit_cfg[EXPRESSION])
            return set(expr_criteria).issubset(resolved_criteria)
        return True  # Age-based criteria have no dependencies

    @staticmethod
    def ensure_all_resolved(unresolved_criteria: list, max_iter: int):
        """
        Ensure no criteria are left unresolved after maximum iterations.
        """
        if unresolved_criteria:
            unresolved_names = [c for c, _ in unresolved_criteria]
            raise ValueError(
                f"Could not resolve criteria after {max_iter} iterations: {unresolved_names}. "
                "Please check for undefined or circular dependencies."
            )


class CriteriaExtraction:
    @staticmethod
    def extract_expression(
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
    def extract_age(
        initial_results: pd.DataFrame, min_age: int = None, max_age: int = None
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

    @staticmethod
    def extract_codes(
        combined_df: pd.DataFrame,
        crit_cfg: dict,
    ) -> pd.DataFrame:
        """
        Fully vectorized extraction for a code/numeric criterion.
        """
        df = combined_df.copy()
        df = CriteriaExtraction._compute_time_window_for_criterion(
            df, crit_cfg.get(START_DAYS), crit_cfg.get(END_DAYS)
        )
        df[TIME_MASK] = compute_time_mask_exclusive(df)
        is_background = df[TIMESTAMP_COL].isna()
        df[TIME_MASK] = df[TIME_MASK] | is_background

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

        has_numeric = NUMERIC_VALUE in crit_cfg or EXTRACT_VALUE in crit_cfg
        if has_numeric:
            min_val = crit_cfg.get(MIN_VALUE)
            max_val = crit_cfg.get(MAX_VALUE)
            extract_val = crit_cfg.get(EXTRACT_VALUE, False)
            res = extract_numeric_values(
                df, flag_df, min_val, max_val, extract_value=extract_val
            )
        else:
            res = flag_df.copy()
            res[NUMERIC_VALUE] = None
        return res

    @staticmethod
    def extract_count_criteria(
        initial_results: pd.DataFrame, crit_cfg: dict
    ) -> pd.DataFrame:
        """
        Check if the count of true flags among specified criteria is within bounds.

        For each patient, this method:
          1. Casts columns in UNIQUE_CRITERIA_LIST to bool.
          2. Sums the true values.
          3. Verifies MIN_COUNT ≤ sum ≤ MAX_COUNT (defaults: 0 and ∞).

        Args:
            initial_results: DataFrame with PID_COL and boolean flag columns.
            crit_cfg: Dict with:
                UNIQUE_CRITERIA_LIST: List of flag column names to count.
                MIN_COUNT (int, optional): Lower inclusive bound (default 0).
                MAX_COUNT (int, optional): Upper inclusive bound (default ∞).

        Returns:
            DataFrame with PID_COL and CRITERION_FLAG (True if within bounds).
        """
        df = initial_results.copy()
        # Ensure booleans
        for c in crit_cfg[UNIQUE_CRITERIA_LIST]:
            df[c] = df[c].astype(bool)
        count = df[crit_cfg[UNIQUE_CRITERIA_LIST]].sum(axis=1)

        low = crit_cfg.get(MIN_COUNT, 0)
        high = crit_cfg.get(MAX_COUNT, float("inf"))
        df[CRITERION_FLAG] = (count >= low) & (count <= high)
        return df[[PID_COL, CRITERION_FLAG]]

    @staticmethod
    def _compute_time_window_for_criterion(
        df: pd.DataFrame, start_days: float = None, end_days: float = None
    ) -> pd.DataFrame:
        """
        Compute time window columns (MIN_TIME and MAX_TIME) based on start_days and end_days.

        Args:
            df: DataFrame containing INDEX_DATE column
            start_days: float, number of days before index date
            end_days: float, number of days after index date

        Returns:
            DataFrame with added MIN_TIME and MAX_TIME columns
        """
        if start_days is not None:
            df[MIN_TIME] = df[INDEX_DATE] + pd.to_timedelta(start_days, unit="D")
        else:
            df[MIN_TIME] = pd.Timestamp.min

        if end_days is not None:
            df[MAX_TIME] = df[INDEX_DATE] + pd.to_timedelta(end_days, unit="D")
        else:
            df[MAX_TIME] = df[INDEX_DATE]

        return df
