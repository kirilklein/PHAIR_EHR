import re

from corebehrt.constants.cohort import (
    ALLOWED_OPERATORS,
    CODE_ENTRY,
    CODE_GROUPS,
    DAYS,
    EXCLUDE_CODES,
    EXPRESSION,
    MAX_AGE,
    MAX_VALUE,
    MIN_AGE,
    MIN_VALUE,
    NUMERIC_VALUE,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    extract_criteria_names_from_expression,
)


def check_criteria_definitions(criteria_definitions: dict) -> None:
    """
    Here we check that each criteria definition is valid.
    Either codes, expression or min_age/max_age must be present.
        Codes should be a list of regexes.
        An expression consists of criteria names separated by | for "OR", & for "AND", ~ for "NOT".
        These criteria names must be present in the criteria_definitions.
    If codes are present, a numeric_value (range) can be defined.
    If numeric_value is present, it must contain min_value or max_value (we take it inclusive).
    """
    criteria_names = list(criteria_definitions.keys())
    for criterion in criteria_names:
        crit_cfg = criteria_definitions[criterion]
        has_codes = CODE_ENTRY in crit_cfg
        has_expression = EXPRESSION in crit_cfg
        has_age_range = MIN_AGE in crit_cfg or MAX_AGE in crit_cfg

        # Check that at least one is present and they are mutually exclusive
        if not (has_codes or has_expression or has_age_range):
            raise ValueError(
                f"Criterion {criterion} must have either codes, expression, or age range (min_age/max_age)"
            )

        # Check that only one type is present
        if sum([has_codes, has_expression, has_age_range]) > 1:
            raise ValueError(
                f"Criterion {criterion} can only have one of: codes, expression, or age range (min_age/max_age). \
                             For complex criteria, use separate criteria definitions and combine them via expression."
            )

        if CODE_ENTRY in crit_cfg:
            codes = crit_cfg[CODE_ENTRY]
            check_codes(codes, criterion)

        if EXCLUDE_CODES in crit_cfg:
            if CODE_ENTRY not in crit_cfg:
                raise ValueError(f"exclude_codes for {criterion} must have codes")
            exclude_codes = crit_cfg[EXCLUDE_CODES]
            check_codes(exclude_codes, criterion)

        if EXPRESSION in crit_cfg:
            expression = crit_cfg[EXPRESSION]
            check_expression(expression, criteria_names)

        check_age(criterion, crit_cfg.get(MIN_AGE), crit_cfg.get(MAX_AGE))

        if NUMERIC_VALUE in crit_cfg:
            if CODE_ENTRY not in crit_cfg:
                raise ValueError(f"numeric_value for {criterion} must have codes")
            numeric_value = crit_cfg[NUMERIC_VALUE]
            check_numeric_value(numeric_value, criterion)


def check_delays_config(delays_config: dict) -> None:
    """Check that delays_config has valid codes (which should be strings) and days (which should be ints)"""
    if len(delays_config) == 0:
        return
    if CODE_GROUPS not in delays_config or DAYS not in delays_config:
        raise ValueError(
            f"delays_config must contain both '{CODE_GROUPS}' and '{DAYS}' keys"
        )
    for code_group in delays_config[CODE_GROUPS]:
        if not isinstance(code_group, str):
            raise ValueError(f"Code group for delays must be a string")
        if not isinstance(delays_config[DAYS], int):
            raise ValueError(f"Days for delays must be an integer")


def check_numeric_value(numeric_value: dict, criterion: str) -> None:
    """Check that numeric_value is valid."""
    if (MIN_VALUE not in numeric_value) and (MAX_VALUE not in numeric_value):
        raise ValueError(
            f"numeric_value for {criterion} must have min_value or max_value"
        )
    min_value = numeric_value.get(MIN_VALUE)
    max_value = numeric_value.get(MAX_VALUE)
    if min_value is not None:
        if not isinstance(min_value, float) and not isinstance(min_value, int):
            raise ValueError(f"min_value for {criterion} must be a float or int")

    if max_value is not None:
        if not isinstance(max_value, float) and not isinstance(max_value, int):
            raise ValueError(f"max_value for {criterion} must be a float or int")

    if min_value is not None and max_value is not None:
        if min_value > max_value:
            raise ValueError(f"min_value for {criterion} must be less than max_value")


def check_age(
    criterion: str,
    min_age: int = None,
    max_age: int = None,
) -> None:
    """Check that min_age and max_age are valid, if present."""
    if min_age is not None:
        if not isinstance(min_age, int) or min_age < 0:
            raise ValueError(f"min_age for {criterion} must be a non-negative integer")

    if max_age is not None:
        if not isinstance(max_age, int) or max_age < 0:
            raise ValueError(f"max_age for {criterion} must be a non-negative integer")
        if min_age is not None:
            if min_age > max_age:
                raise ValueError(f"min_age for {criterion} must be less than max_age")


def check_codes(codes: list, criterion: str) -> None:
    """Check that codes are valid. Are strings and valid regexes."""
    for code in codes:
        if not isinstance(code, str):
            raise ValueError(f"Code for {criterion} must be a string")
        try:
            re.compile(code)
        except re.error:
            raise ValueError(f"Code for {criterion} must be a valid regular expression")


def is_valid_criterion_name(name: str) -> bool:
    """
    Returns True if the criterion name contains only letters, digits, underscores, or forward slashes.
    """
    # Define allowed pattern: alphanumeric characters, underscore, or slash.
    return bool(re.match(r"^[A-Za-z0-9_/]+$", name))


def check_expression(expression: str, criteria_names: list) -> None:
    """
    An expression consists of criterion names separated by operators (|, &, ~,
    or their word equivalents: and, or, not). These criterion names must:
      - appear in the criteria_definitions, and
      - be composed only of allowed characters (letters, digits, underscores, and slashes).

    Raises:
        ValueError: if any token is not permitted.
    """
    # Ensure the expression contains at least one operator.
    if not any(op in expression for op in ALLOWED_OPERATORS):
        raise ValueError(
            f"Expression '{expression}' must contain at least one operator (|, &, ~, and, or, not)."
        )

    allowed_names = set(criteria_names)
    criteria_in_expr = extract_criteria_names_from_expression(expression)

    # Check that every extracted criterion name is in the allowed names.
    unknown_criteria = [c for c in criteria_in_expr if c not in allowed_names]
    if unknown_criteria:
        raise ValueError(f"Unknown criteria in expression: {unknown_criteria}")

    # Additionally, ensure that each criterion name matches the allowed pattern.
    for name in criteria_in_expr:
        if not is_valid_criterion_name(name):
            raise ValueError(
                f"Criterion name '{name}' contains invalid characters. "
                "Only letters, digits, underscores, and forward slashes are allowed."
            )


def check_criteria_names(df, criteria_names):
    """Check if all criteria names are present in the DataFrame."""
    missing_criteria = set(criteria_names) - set(df.columns)
    if missing_criteria:
        raise ValueError(f"Criteria not found in DataFrame: {missing_criteria}")
