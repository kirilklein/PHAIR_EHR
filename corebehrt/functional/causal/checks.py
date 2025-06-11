import pandas as pd


def check_time_windows(time_windows: dict):
    """
    Check that the time windows config dict is valid.
    Requires data_end, data_start, min_follow_up, min_lookback.
    """
    if "data_end" not in time_windows:
        raise ValueError("data_end must be provided")
    if "data_start" not in time_windows:
        raise ValueError("data_start must be provided")

    try:
        data_end = pd.Timestamp(**time_windows["data_end"])
    except KeyError as e:
        raise ValueError("data_end must be provided as year, month, day") from e
    try:
        data_start = pd.Timestamp(**time_windows["data_start"])
    except KeyError as e:
        raise ValueError("data_start must be provided as year, month, day") from e

    if data_end < data_start:
        raise ValueError("data_end must be greater than data_start")

    if "min_follow_up" not in time_windows:
        raise ValueError("min_follow_up must be provided")
    try:
        pd.Timedelta(**time_windows["min_follow_up"])
    except KeyError as e:
        raise ValueError(
            "min_follow_up can be given in weeks, days, seconds, minutes, hours"
        ) from e
    if "min_lookback" not in time_windows:
        raise ValueError(
            "min_lookback can be given in weeks, days, seconds, minutes, hours, or years"
        )
    try:
        pd.Timedelta(**time_windows["min_lookback"])
    except KeyError as e:
        raise ValueError(
            "min_lookback can be given in weeks, days, seconds, minutes, hours, or years"
        ) from e
