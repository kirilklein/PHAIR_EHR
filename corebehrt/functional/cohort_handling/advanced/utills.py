from typing import Dict
from corebehrt.constants.cohort import (
    EXCLUDED_BY_INCLUSION_CRITERIA,
    EXCLUDED_BY_EXCLUSION_CRITERIA,
    FINAL_INCLUDED,
    N_EXCLUDED_BY_CODE_LIMITS,
)


def print_stats(stats: Dict):
    print("----- Criteria Application Summary -----")
    print(f"Initial total: {stats['initial_total']}")
    print("\nExcluded by individual criteria:")
    for k, v in stats[EXCLUDED_BY_INCLUSION_CRITERIA].items():
        print(f"  - {k}: {v}")
    if stats[EXCLUDED_BY_EXCLUSION_CRITERIA]:
        print("\nExcluded by unique code limits:")
        for k, v in stats[N_EXCLUDED_BY_CODE_LIMITS].items():
            print(f"  - {k}: {v}")
    print(f"\nFinal included: {stats[FINAL_INCLUDED]}")
    print("----------------------------------------")
