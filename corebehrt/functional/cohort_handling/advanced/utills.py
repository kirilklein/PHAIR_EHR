from typing import Dict
from corebehrt.constants.cohort import (
    EXCLUDED_BY_INCLUSION_CRITERIA,
    EXCLUDED_BY_EXCLUSION_CRITERIA,
    FINAL_INCLUDED,
)


def print_stats(stats: Dict):
    print("----- Criteria Application Summary -----")
    print(f"Initial total: {stats['initial_total']}")

    print("\nExcluded by individual inclusion criteria:")
    for k, v in stats[EXCLUDED_BY_INCLUSION_CRITERIA].items():
        print(f"  - {k}: {v}")

    print("\nExcluded by individual exclusion criteria:")
    for k, v in stats[EXCLUDED_BY_EXCLUSION_CRITERIA].items():
        print(f"  - {k}: {v}")

    print(f"\nFinal included: {stats[FINAL_INCLUDED]}")
    print("----------------------------------------")
