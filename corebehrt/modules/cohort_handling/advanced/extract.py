from typing import Dict, List
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
    MAX_AGE,
    MIN_AGE,
    NUMERIC_VALUE,
    TIME_MASK,
    TIME_WINDOW_DAYS,
    INDEX_DATE,
    MIN_VALUE,
    MAX_VALUE,
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
    merge_index_dates,
    rename_result,
    extract_numeric_values,
)


# --- Main Extraction Function ---
def extract_patient_criteria(
    events: pd.DataFrame,
    index_dates: pd.DataFrame,
    criteria_definitions: dict,
    delays_config: dict = None,
) -> pd.DataFrame:
    """
    Extract and evaluate inclusion/exclusion criteria for each patient in a vectorized fashion.

    This function processes "simple" criteria (code- and numeric-based) first, then computes
    additional criteria:
      - Composite criteria defined via an expression (EXPRESSION key)
      - Age-based criteria (using MIN_AGE and/or MAX_AGE)

    Also computes the patient's age at index_date and merges it into the final results.

    Returns a DataFrame with one row per patient (subject_id) and one column per criterion.
    Additionally, an "age_at_index_date" column is added.
    """
    if delays_config is None:
        delays_config = {}

    check_delays_config(delays_config)
    check_criteria_definitions(criteria_definitions)

    shard_ids = events[PID_COL].unique()
    relevant_index_dates = index_dates[index_dates[PID_COL].isin(shard_ids)]
    relevant_index_dates = relevant_index_dates.rename(
        columns={TIMESTAMP_COL: INDEX_DATE}
    )
    print(f"Processing {len(relevant_index_dates)} patients")

    # Compute age at index_date from events (extracting DOB events)
    age_df = compute_age_at_index_date(relevant_index_dates, events)

    # Partition criteria:
    simple_criteria = []
    additional_criteria = []  # Composite or age-based

    for criterion, crit_cfg in criteria_definitions.items():
        if CODE_ENTRY in crit_cfg:
            simple_criteria.append((criterion, crit_cfg))
        elif (MIN_AGE in crit_cfg) or (MAX_AGE in crit_cfg) or (EXPRESSION in crit_cfg):
            additional_criteria.append((criterion, crit_cfg))

    # Process simple (code/numeric) criteria: collect results as a list of DataFrames.
    simple_results_list: List[pd.DataFrame] = []
    for criterion, crit_cfg in simple_criteria:
        res = vectorized_extraction_codes(
            events, relevant_index_dates, crit_cfg, delays_config
        )
        has_numeric = NUMERIC_VALUE in crit_cfg
        # Rename output columns and select only the desired ones.
        res = rename_result(res, criterion, has_numeric)
        simple_results_list.append(res)

    if simple_results_list:
        simple_results = simple_results_list[0]
        for df in simple_results_list[1:]:
            simple_results = simple_results.merge(df, on=PID_COL, how="outer")
    else:
        simple_results = pd.DataFrame({PID_COL: relevant_index_dates[PID_COL].unique()})

    # Merge age information.
    all_results = simple_results.merge(age_df, on=PID_COL, how="left")

    # Process additional criteria.
    additional_results_list: List[pd.DataFrame] = []
    for criterion, crit_cfg in additional_criteria:
        if EXPRESSION in crit_cfg:
            res = vectorized_extraction_expression(crit_cfg[EXPRESSION], all_results)
        elif (MIN_AGE in crit_cfg) or (MAX_AGE in crit_cfg):
            res = vectorized_extraction_age(
                all_results, crit_cfg.get(MIN_AGE), crit_cfg.get(MAX_AGE)
            )
        else:
            continue
        # Rename and select only PID_COL and the new criterion flag.
        res = res.rename(columns={CRITERION_FLAG: criterion})[[PID_COL, criterion]]
        additional_results_list.append(res)

    if additional_results_list:
        additional_results = additional_results_list[0]
        for df in additional_results_list[1:]:
            additional_results = additional_results.merge(df, on=PID_COL, how="outer")
        all_results = all_results.merge(additional_results, on=PID_COL, how="left")

    return all_results


# --- Vectorized Extraction Functions ---
def vectorized_extraction_codes(
    events: pd.DataFrame,
    index_dates: pd.DataFrame,
    crit_cfg: Dict,
    delays_config: Dict,
) -> pd.DataFrame:
    """
    Fully vectorized extraction for a code/numeric criterion.

    Steps:
      1. Merge index_dates into events.
      2. Compute per-event delay using delays_config.
      3. Compute the time window (min_timestamp and max_timestamp).
      4. Build time and code masks and combine them into FINAL_MASK.
      5. Group by patient: a patient’s CRITERION_FLAG is True if any event passes FINAL_MASK.
      6. If numeric extraction is requested, additionally extract, for each patient, the latest numeric_value
         (subject to additional range filtering on min_value and/or max_value).

    Returns a DataFrame with columns: PID_COL, CRITERION_FLAG, and (if applicable) NUMERIC_VALUE.
    """
    df = merge_index_dates(events, index_dates)
    # Compute delay column; note that here we pass delays_config's CODE_GROUPS and DAYS values.
    df = compute_delay_column(
        df, delays_config.get(CODE_GROUPS, []), delays_config.get(DAYS, 0)
    )
    # Compute time window columns – use a provided time window or default to 36500 days.
    df = compute_time_window_columns(df, crit_cfg.get(TIME_WINDOW_DAYS, 36500))

    df[TIME_MASK] = compute_time_mask(df)
    # Build allowed code mask, using crit_cfg[CODE_ENTRY] and optional crit_cfg[EXCLUDE_CODES].
    df[CODE_MASK] = compute_code_masks(
        df, crit_cfg[CODE_ENTRY], crit_cfg.get(EXCLUDE_CODES, [])
    )
    df[FINAL_MASK] = df[TIME_MASK] & df[CODE_MASK]

    # Group by patient: the flag is True if at least one event passes FINAL_MASK.
    flag_df = (
        df.groupby(PID_COL)[FINAL_MASK]
        .any()
        .reset_index()
        .rename(columns={FINAL_MASK: CRITERION_FLAG})
    )

    if NUMERIC_VALUE in crit_cfg:
        min_val = crit_cfg.get(MIN_VALUE, None)
        max_val = crit_cfg.get(MAX_VALUE, None)
        result = extract_numeric_values(df, flag_df, min_val, max_val)
    else:
        result = flag_df.copy()
        result[NUMERIC_VALUE] = None
    return result


def vectorized_extraction_expression(
    expression: str, initial_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate a composite criterion defined by an expression.

    Returns a DataFrame with columns:
       PID_COL and CRITERION_FLAG.
    """
    local_dict = {
        col.upper(): initial_results[col]
        for col in initial_results.columns
        if col != PID_COL
    }
    composite_flag = pd.eval(expression, engine="python", local_dict=local_dict)
    result = initial_results[[PID_COL]].copy()
    result[CRITERION_FLAG] = composite_flag
    return result


def vectorized_extraction_age(
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
