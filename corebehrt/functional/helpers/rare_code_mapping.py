"""
Functions for mapping rare codes to more common parent codes or group labels.
Rare codes (counts below threshold) are processed as:
- Hierarchical codes (matching regex): trim number part after separator
- Non-hierarchical codes: map to "<group>/rare"
"""

import re
import warnings
from typing import Dict, List

from corebehrt.constants.helper import MIN_DETAIL_LENGTH, RARE_STR


def group_rare_codes(
    counts: Dict[str, int],
    rare_threshold: int,
    hierarchical_pattern: str,
    group_separator: str = "/",
) -> Dict[str, str]:
    """
    Maps rare codes to their aggregated form, processing longest codes first.
    """
    if not group_separator:
        warnings.warn("Empty group separator provided, defaulting to '/'")
        group_separator = "/"

    aggregated_counts = counts.copy()
    mapping = {code: code for code in counts}

    i = 0
    while should_continue_aggregation(aggregated_counts, rare_threshold, i):
        changed = False
        rare_codes = find_rare_codes(aggregated_counts, rare_threshold)
        rare_codes.sort(key=len, reverse=True)

        for code in rare_codes:
            # Skip if:
            # - code was already processed
            # - is no longer rare
            # - already contains RARE_STR
            if (
                code not in aggregated_counts
                or aggregated_counts[code] >= rare_threshold
                or RARE_STR in code
            ):
                continue

            hierarchical = is_hierarchical(code, hierarchical_pattern)
            new_code = get_new_code(code, hierarchical, group_separator)
            if new_code == code:
                continue

            aggregated_counts[new_code] = (
                aggregated_counts.get(new_code, 0) + aggregated_counts[code]
            )
            del aggregated_counts[code]

            for orig_code, current in mapping.items():
                if current == code:
                    mapping[orig_code] = new_code
            changed = True
        i += 1

        if not changed:
            break

    return mapping


def get_new_code(
    code: str, is_hierarchical_flag: bool, group_separator: str = "/"
) -> str:
    """
    Get aggregated form of a code.
    - Hierarchical with separator: trim detail part until MIN_DETAIL_LENGTH, then map to "<group>/rare"
    - Hierarchical without separator: trim until MIN_DETAIL_LENGTH, then map to "<first_char>/rare"
    - Non-hierarchical: always map to "<first_part>/rare"
    """
    # Don't process codes that are already rare
    if RARE_STR in code:
        return code

    if not code:
        return code

    parts = code.split(group_separator)

    if is_hierarchical_flag:
        return (
            handle_hierarchical_with_separator(parts[0], parts[1], group_separator)
            if len(parts) > 1
            else handle_hierarchical_without_separator(code, group_separator)
        )

    return handle_non_hierarchical(code, group_separator)


def find_rare_codes(counts: Dict[str, int], rare_threshold: int) -> List[str]:
    """Find codes with counts below threshold."""
    return [code for code, cnt in counts.items() if cnt < rare_threshold]


def get_group_label(code: str, group_separator: str = "/") -> str:
    """Extract the group part before the separator."""
    parts = code.split(group_separator)
    return parts[0] if parts else code[0]


def is_hierarchical(code: str, hierarchical_pattern: str) -> bool:
    """Check if code matches the hierarchical pattern."""
    if not code:
        return False
    return bool(re.search(hierarchical_pattern, code))


def handle_hierarchical_with_separator(group: str, detail: str, separator: str) -> str:
    """Process hierarchical code that contains a separator."""
    if len(detail) > MIN_DETAIL_LENGTH:
        return f"{group}{separator}{detail[:-1]}"
    return f"{group}{separator}{RARE_STR}"


def handle_hierarchical_without_separator(code: str, separator: str) -> str:
    """Process hierarchical code without separator."""
    if len(code) > MIN_DETAIL_LENGTH:
        return code[:-1]
    return f"{code[0]}{separator}{RARE_STR}"


def handle_non_hierarchical(code: str, separator: str) -> str:
    """Process non-hierarchical code, taking first part or character."""
    parts = code.split(separator)
    group = parts[0] if len(parts) > 1 else code[0]
    return f"{group}{separator}{RARE_STR}"


def should_continue_aggregation(
    counts: Dict[str, int], rare_threshold: int, i: int
) -> bool:
    """Check if further aggregation is needed."""
    max_iterations = 8  # should be enough for all hierarchical codes
    return any(cnt < rare_threshold for cnt in counts.values()) and i < max_iterations
