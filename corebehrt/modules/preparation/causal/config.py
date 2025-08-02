from dataclasses import dataclass
import numpy as np


@dataclass
class ExposureConfig:
    """ "A container holding the exposureconfig attributes"""

    n_hours_start_follow_up: int = -1
    n_hours_end_follow_up: int = np.inf
    n_hours_censoring: int = -1


@dataclass
class OutcomeConfig:
    """ "A container holding the exposureconfig attributes"""

    n_hours_start_follow_up: int
    n_hours_end_follow_up: int = np.inf
    n_hours_compliance: int = np.inf
    group_wise_follow_up: bool = False
    min_instances_per_class: int = 10
    delay_death_hours: int = 0
