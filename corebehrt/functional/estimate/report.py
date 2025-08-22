import pandas as pd
from CausalEstimate.stats.stats import compute_treatment_outcome_table

from corebehrt.constants.causal.data import EXPOSURE_COL, OUTCOME, STATUS


def convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
    """
    Convert a dictionary of effects to a pandas DataFrame.

    Takes a dictionary where keys are method names and values are effect estimates,
    and converts it to a DataFrame with a 'method' column and the corresponding values.

    Args:
        effect (dict): Dictionary with method names as keys and effect estimates as values.

    Returns:
        pd.DataFrame: DataFrame with 'method' column and effect values.

    Example:
        >>> effect_dict = {
        ...     'IPW': 0.25,
        ...     'TMLE': 0.23,
        ... }
        >>> df = convert_effect_to_dataframe(effect_dict)
        >>> print(df)
                 method  effect
        0          IPW  0.25
        1          TMLE  0.23
    """
    return (
        pd.DataFrame.from_dict(effect, orient="index")
        .reset_index()
        .rename(columns={"index": "method"})
    )


def compute_outcome_stats(analysis_df: pd.DataFrame, outcome_name: str) -> pd.DataFrame:
    """
    Compute treatment-outcome statistics for a specific outcome.

    This function generates a cross-tabulation of treatment exposure and outcome status,
    providing counts for treated/untreated groups by outcome presence/absence.
    The result includes totals and is tagged with the outcome name for identification
    in multi-outcome analyses.

    Args:
        analysis_df: Analysis cohort dataframe with EXPOSURE_COL and OUTCOME columns
        outcome_name: Name of the outcome being analyzed (e.g., 'diabetes', 'hypertension')

    Returns:
        DataFrame with treatment-outcome statistics including:
        - status: Treatment group ('Untreated', 'Treated', 'Total')
        - No Outcome: Count of individuals without the outcome
        - Outcome: Count of individuals with the outcome
        - Total: Total count for each group
        - outcome: Column tagged with the provided outcome_name

    Example:
        >>> analysis_df = pd.DataFrame({
        ...     'exposure': [1, 1, 0, 0, 1, 0, 0, 1],
        ...     'outcome': [1, 0, 1, 0, 1, 0, 1, 0]
        ... })
        >>> stats = compute_outcome_stats(analysis_df, 'diabetes')
        >>> print(stats)
           status  No Outcome  Outcome  Total   outcome
        0  Untreated         2        2      4  diabetes
        1    Treated         2        2      4  diabetes
        2      Total         4        4      8  diabetes
    """
    stats_table = compute_treatment_outcome_table(analysis_df, EXPOSURE_COL, OUTCOME)
    stats_table = stats_table.reset_index(drop=False)
    stats_table.rename(columns={"index": STATUS}, inplace=True)
    stats_table[OUTCOME] = outcome_name
    return stats_table
