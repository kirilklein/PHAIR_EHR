import re
from typing import Any, Dict, List

from corebehrt.constants.cohort import (
    ALLOWED_OPERATORS,
    CODE_ENTRY,
    EXCLUDE_CODES,
    EXPRESSION,
    MAX_AGE,
    MAX_COUNT,
    MAX_VALUE,
    MIN_AGE,
    MIN_COUNT,
    MIN_VALUE,
    NUMERIC_VALUE,
    UNIQUE_CRITERIA_LIST,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    extract_criteria_names_from_expression,
)


class CriteriaValidator:
    """
    Validates cohort criteria definitions for proper structure and values.

    Each criterion must have exactly one of:
    - codes (with optional numeric_value)
    - expression (combining other criteria)
    - age range (min_age/max_age)
    - unique_criteria_list (with min_count/max_count)

    Validates structure, regex patterns, numeric ranges, expressions, age ranges,
    and count-based criteria bounds.
    """

    def __init__(self, definitions: Dict[str, Dict[str, Any]]):
        self.defs = definitions
        self.names = list(definitions.keys())

    def validate(self) -> None:
        """
        Validate all criteria definitions.

        Checks structure, code patterns, numeric values, expressions,
        age ranges, and count-based criteria.

        Raises:
            ValueError: If any validation fails
        """
        for name, cfg in self.defs.items():
            self._validate_exclusivity(name, cfg)
            if CODE_ENTRY in cfg:
                self._validate_codes(name, cfg[CODE_ENTRY])
                if EXCLUDE_CODES in cfg:
                    self._validate_codes(name, cfg[EXCLUDE_CODES], exclude=True)
                if NUMERIC_VALUE in cfg:
                    self._validate_numeric(name, cfg[NUMERIC_VALUE])

            if EXPRESSION in cfg:
                self.validate_expression(name, cfg[EXPRESSION])

            if MIN_AGE in cfg or MAX_AGE in cfg:
                self._validate_age(name, cfg.get(MIN_AGE), cfg.get(MAX_AGE))

            if UNIQUE_CRITERIA_LIST in cfg:
                self._validate_count(name, cfg)

    def _validate_exclusivity(self, name: str, cfg: Dict[str, Any]) -> None:
        """
        Ensure criterion has exactly one type of definition (code, expression, age, or count).

        Raises:
            ValueError: If criterion has no definition or multiple definitions
        """
        flags = {
            "code": CODE_ENTRY in cfg,
            "expression": EXPRESSION in cfg,
            "age": MIN_AGE in cfg or MAX_AGE in cfg,
            "count": UNIQUE_CRITERIA_LIST in cfg,
        }
        if not any(flags.values()):
            raise ValueError(
                f"'{name}' must define one of: code, expression, age, or count-based rule."
            )
        if sum(flags.values()) > 1:
            raise ValueError(
                f"'{name}' defines multiple rule types {list(f for f, v in flags.items() if v)}; only one allowed."
            )

    def _validate_codes(
        self, name: str, codes: List[str], exclude: bool = False
    ) -> None:
        """
        Validate code patterns are non-empty list of valid regex strings.

        Raises:
            ValueError: If codes are invalid
        """
        if not isinstance(codes, list) or not codes:
            kind = "exclude_codes" if exclude else CODE_ENTRY
            raise ValueError(
                f"'{name}' {kind} must be a non-empty list of regex strings."
            )
        for pattern in codes:
            if not isinstance(pattern, str):
                raise ValueError(f"'{name}': code patterns must be strings.")
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(f"'{name}': invalid regex '{pattern}'.")

    def _validate_numeric(self, name: str, nv: Dict[str, Any]) -> None:
        """
        Validate numeric value configuration has valid min/max bounds.

        Raises:
            ValueError: If numeric configuration is invalid
        """
        if not isinstance(nv, dict) or (MIN_VALUE not in nv and MAX_VALUE not in nv):
            raise ValueError(
                f"'{name}': numeric_value requires min_value or max_value."
            )
        lo, hi = nv.get(MIN_VALUE), nv.get(MAX_VALUE)
        for val, key in ((lo, MIN_VALUE), (hi, MAX_VALUE)):
            if val is not None and not isinstance(val, (int, float)):
                raise ValueError(f"'{name}': {key} must be a number.")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"'{name}': min_value > max_value.")

    def validate_expression(self, name: str, expr: str) -> None:
        """
        Validate expression has valid operators and references existing criteria.

        Raises:
            ValueError: If expression is invalid
        """
        crits = extract_criteria_names_from_expression(expr)
        if len(crits) > 1 and not any(op in expr for op in ALLOWED_OPERATORS):
            raise ValueError(f"'{name}': expression '{expr}' needs an operator.")
        invalid = [c for c in crits if c not in self.names]
        if invalid:
            raise ValueError(f"'{name}': unknown criteria in expression: {invalid}")
        for token in crits:
            if not re.match(r"^[A-Za-z0-9_/]+$", token):
                raise ValueError(f"'{name}': invalid chars in criterion '{token}'")

    def _validate_age(self, name: str, min_age: Any, max_age: Any) -> None:
        """
        Validate age range has valid non-negative bounds.

        Raises:
            ValueError: If age configuration is invalid
        """
        for val, label in ((min_age, MIN_AGE), (max_age, MAX_AGE)):
            if val is not None:
                if not isinstance(val, int) or val < 0:
                    raise ValueError(f"'{name}': {label} must be a non-negative int.")
        if min_age is not None and max_age is not None and min_age > max_age:
            raise ValueError(f"'{name}': min_age > max_age.")

    def _validate_count(self, name: str, cfg: Dict[str, Any]) -> None:
        """
        Validate count-based criteria has valid bounds and existing sub-criteria.

        Raises:
            ValueError: If count configuration is invalid
        """
        if (MAX_COUNT not in cfg) and (MIN_COUNT not in cfg):
            raise ValueError(f"'{name}': count rules need MAX_COUNT or MIN_COUNT.")
        ulist = cfg[UNIQUE_CRITERIA_LIST]
        if not isinstance(ulist, list) or not ulist:
            raise ValueError(
                f"'{name}': UNIQUE_CRITERIA_LIST must be a non-empty list."
            )
        unknown = [c for c in ulist if c not in self.names]
        if unknown:
            raise ValueError(f"'{name}': unknown sub-criteria: {unknown}")

        if name in ulist:
            raise ValueError(
                f"'{name}': cannot reference itself in UNIQUE_CRITERIA_LIST"
            )

        lo, hi = cfg.get(MIN_COUNT, 0), cfg.get(MAX_COUNT, int(1e6))
        for val, label in ((lo, MIN_COUNT), (hi, MAX_COUNT)):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"'{name}': {label} must be a non-negative int.")
        if lo > hi:
            raise ValueError(f"'{name}': MIN_COUNT > MAX_COUNT.")
