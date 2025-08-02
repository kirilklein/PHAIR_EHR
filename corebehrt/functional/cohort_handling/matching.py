import pandas as pd
from typing import List
from typing import Literal
import re


def get_col_booleans(
    concepts_plus: pd.DataFrame,
    columns: List,
    patterns: List[List[str]],
    match_how: Literal["startswith", "contains", "exact"] = "startswith",
    case_sensitive: bool = True,
) -> list:
    """
    Get boolean columns for each type and match.
    """
    col_booleans = []
    for col, pattern in zip(columns, patterns):
        if match_how == "startswith":
            col_bool = startswith_match(concepts_plus, col, pattern, case_sensitive)
        elif match_how == "contains":
            col_bool = contains_match(concepts_plus, col, pattern, case_sensitive)
        elif match_how == "exact":
            col_bool = exact_match(concepts_plus, col, pattern, case_sensitive)
        else:
            raise ValueError(
                f"match_how must be 'startswith', 'contains', or 'exact', not '{match_how}'"
            )
        col_booleans.append(col_bool)
    return col_booleans


def startswith_match(
    df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool
) -> pd.Series:
    """Match strings using startswith"""
    if not case_sensitive:
        patterns = [x.lower() for x in patterns]
        return df[column].astype(str).str.lower().str.startswith(tuple(patterns), False)
    return df[column].astype(str).str.startswith(tuple(patterns), False)


def contains_match(
    df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool
) -> pd.Series:
    """Match strings using a single vectorized regex 'contains' call."""
    if not patterns:
        return pd.Series([False] * len(df), index=df.index)

    # Escape patterns to treat special characters literally, then join with '|'
    regex_pattern = "|".join(re.escape(p) for p in patterns)

    # Perform a single, fast, vectorized call
    return (
        df[column]
        .astype(str)
        .str.contains(regex_pattern, case=case_sensitive, regex=True, na=False)
    )


def exact_match(
    df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool
) -> pd.Series:
    """Match strings using exact match (optimized for categorical)"""
    # Assumes df[column] is already a categorical dtype
    if not case_sensitive:
        patterns = [x.lower() for x in patterns]
        # Perform .lower() only if necessary. Note: This part is still slow.
        return df[column].str.lower().isin(patterns)

    # This is the highly optimized path for case-sensitive exact matches
    return df[column].isin(patterns)
