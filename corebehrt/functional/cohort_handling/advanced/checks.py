import re

from corebehrt.constants.cohort import (
    ALLOWED_OPERATORS,
    CODE_ENTRY,
    CRITERIA,
    EXCLUDE_CODES,
    EXPRESSION,
    MAX_AGE,
    MAX_COUNT,
    MIN_COUNT,
    MAX_VALUE,
    MIN_AGE,
    MIN_VALUE,
    NUMERIC_VALUE,
    UNIQUE_CRITERIA_LIST,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    extract_criteria_names_from_expression,
)


def check_criteria_definitions(criteria_definitions: dict) -> None:
    """
    Here we check that each criteria definition is valid.
    Either codes, unique_criteria_list, expression or min_age/max_age must be present.
        Codes should be a list of regexes.
        Unique_criteria_list should be a list of criteria names.
        An expression consists of criteria names separated by | for "OR", & for "AND", ~ for "NOT".
        These criteria names must be present in the criteria_definitions.
    If codes are present, a numeric_value (range) can be defined.
    If unique_criteria_list is present, min_count or max_count must be defined.
    If numeric_value is present, it must contain min_value or max_value (we take it inclusive).
    """
    criteria_names = list(criteria_definitions.keys())
    for criterion in criteria_names:
        crit_cfg = criteria_definitions[criterion]
        has_codes = CODE_ENTRY in crit_cfg
        has_expression = EXPRESSION in crit_cfg
        has_age_range = MIN_AGE in crit_cfg or MAX_AGE in crit_cfg
        has_min_count = MIN_COUNT in crit_cfg
        has_max_count = MAX_COUNT in crit_cfg
        has_unique_criteria_list = UNIQUE_CRITERIA_LIST in crit_cfg

        # Check that at least one is present and they are mutually exclusive
        if not (
            has_codes or has_expression or has_age_range or has_unique_criteria_list
        ):
            raise ValueError(
                f"Criterion '{criterion}' must have exactly one of: "
                "codes, expression, age range, or unique_criteria_list."
            )
        # Check that only one type is present
        if (
            sum([has_codes, has_expression, has_age_range, has_unique_criteria_list])
            > 1
        ):
            raise ValueError(
                f"Criterion {criterion} can only have one of: codes, expression, or age range (min_age/max_age). \
                             For complex criteria, use separate criteria definitions and combine them via expression."
            )

        if has_codes:
            codes = crit_cfg[CODE_ENTRY]
            check_codes(codes, criterion)

        if EXCLUDE_CODES in crit_cfg:
            if not has_codes:
                raise ValueError(f"exclude_codes for {criterion} must have codes")
            exclude_codes = crit_cfg[EXCLUDE_CODES]
            check_codes(exclude_codes, criterion)

        if has_unique_criteria_list:
            unique_criteria_list = crit_cfg[UNIQUE_CRITERIA_LIST]
            if (not has_min_count) and (not has_max_count):
                raise ValueError(
                    f"Criteria with unique_criteria_list must have min_count or max_count"
                )
            check_unique_criteria_list(
                unique_criteria_list,
                criterion,
                criteria_names,
                crit_cfg.get(MIN_COUNT),
                crit_cfg.get(MAX_COUNT),
            )

        if has_expression:
            expression = crit_cfg[EXPRESSION]
            check_expression(expression, criteria_names)

        check_age(criterion, crit_cfg.get(MIN_AGE), crit_cfg.get(MAX_AGE))

        if NUMERIC_VALUE in crit_cfg:
            if not has_codes:
                raise ValueError(f"numeric_value for {criterion} must have codes")
            numeric_value = crit_cfg[NUMERIC_VALUE]
            check_numeric_value(numeric_value, criterion)


def check_unique_criteria_list(
    unique_criteria_list: list,
    criterion: str,
    criteria_names: list,
    min_count: int = None,
    max_count: int = None,
) -> None:
    """Check that unique_criteria_list is valid."""
    if not isinstance(unique_criteria_list, list):
        raise ValueError(f"unique_criteria_list for {criterion} must be a list")

    if min_count is not None:
        if not isinstance(min_count, int) or min_count < 0:
            raise ValueError(
                f"min_count for {criterion} must be a non-negative integer"
            )
        if min_count > len(unique_criteria_list):
            raise ValueError(
                f"min_count for {criterion} must be less than or equal to the number of criteria in unique_criteria_list"
            )
    if max_count is not None:
        if not isinstance(max_count, int) or max_count < 0:
            raise ValueError(
                f"max_count for {criterion} must be a non-negative integer"
            )
    if min_count is not None and max_count is not None:
        if min_count >= max_count:
            raise ValueError(f"min_count for {criterion} must be less than max_count")

    if not unique_criteria_list:
        raise ValueError(f"unique_criteria_list for '{criterion}' cannot be empty")

    for sub_criterion in unique_criteria_list:
        if sub_criterion not in set(criteria_names):
            raise ValueError(
                f"unique_criteria_list for {criterion} must be a list of valid criterion names"
            )


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
    # Pull out all criterion names from the expression
    criteria_in_expr = extract_criteria_names_from_expression(expression)

    # If there’s more than one criterion, ensure at least one operator is present
    if len(criteria_in_expr) > 1 and not any(
        op in expression for op in ALLOWED_OPERATORS
    ):
        raise ValueError(
            f"Composite expression '{expression}' must contain at least one operator "
            f"(|, &, ~, and, or, not)."
        )

    allowed_names = set(criteria_names)

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


def check_unique_code_limits(unique_code_limits: dict, criteria_names: list):
    """
    Check that unique code limits configuration is valid.

    Args:
        unique_code_limits (dict): Dictionary containing code limit configurations.
            Each configuration must have:
            - max_count: Maximum number of codes allowed
            - criteria: List of criterion names to check against
        criteria_names (list): List of valid criterion names to validate against

    Raises:
        ValueError: If configuration is invalid:
            - Missing required fields (max_count or criteria)
            - criteria is not a list
            - criteria contains unknown criterion names
    """
    for name, cfg in unique_code_limits.items():
        if cfg.get(MAX_COUNT) is None:
            raise ValueError(f"Max count must be specified for {name}")
        if cfg.get(CRITERIA) is None:
            raise ValueError(f"Criteria must be specified for {name}")
        if not isinstance(cfg[CRITERIA], list):
            raise ValueError(f"Criteria for {name} must be a list")
        missing_criteria = [c for c in cfg[CRITERIA] if c not in criteria_names]
        if missing_criteria:
            raise ValueError(
                f"The following criteria were not found in criteria_names: {missing_criteria}"
            )
