import itertools
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from corebehrt.constants.causal.data import (
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
)
from corebehrt.functional.estimate.bias import apply_logit_bias, apply_sharpening_bias


@dataclass
class BiasTypes:
    additive = "additive"
    sharpen = "sharpen"


@dataclass
class BiasConfig:
    ps_bias_type: BiasTypes
    y_bias_type: BiasTypes
    ps_values: List[float]
    y_values: List[float]


class BiasIntroducer:
    def __init__(self, bias_config: BiasConfig):
        self.bias_config = bias_config
        self.ps_bias_type = bias_config.ps_bias_type
        self.y_bias_type = bias_config.y_bias_type
        self.ps_values = bias_config.ps_values
        self.y_values = bias_config.y_values

    def get_bias_grid(self) -> List[Tuple[float, float]]:
        """Creates a grid of all combinations of ps and y bias values."""
        return list(itertools.product(self.ps_values, self.y_values))

    def apply_bias(
        self, df: pd.DataFrame, ps_bias: float, y_bias: float
    ) -> pd.DataFrame:
        """Applies specified bias to a dataframe."""
        df_biased = df.copy()

        # Apply bias to propensity score
        if self.ps_bias_type == BiasTypes.additive:
            df_biased[PS_COL] = apply_logit_bias(df_biased[PS_COL], ps_bias)
        elif self.ps_bias_type == BiasTypes.sharpen:
            df_biased[PS_COL] = apply_sharpening_bias(df_biased[PS_COL], ps_bias)

        # Apply bias to outcome probabilities
        outcome_proba_cols = [
            PROBAS,
            PROBAS_EXPOSED,
            PROBAS_CONTROL,
        ]  # apply to true probas and counterfactual probas
        for col in outcome_proba_cols:
            if col in df_biased.columns:
                if self.y_bias_type == BiasTypes.additive:
                    df_biased[col] = apply_logit_bias(df_biased[col], y_bias)
                elif self.y_bias_type == BiasTypes.sharpen:
                    df_biased[col] = apply_sharpening_bias(df_biased[col], y_bias)

        return df_biased
