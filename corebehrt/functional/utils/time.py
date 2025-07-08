from datetime import datetime
from typing import Union

import pandas as pd


def get_hours_since_epoch(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([], dtype=float)
        # Convert timestamps to UTC (timezone-aware)
        timestamps = pd.to_datetime(
            timestamps, utc=True
        )  # ensure consistency across dataset
        # Remove the timezone information to get a timezone-naive series, necessary for the next step
        timestamps = timestamps.dt.tz_localize(None)
        # Cast to microsecond precision
        timestamps = timestamps.astype("datetime64[us]")
        # Convert microseconds to hours
        hours = (timestamps.astype("int64") // 10**6) / 3600
        return hours

    elif isinstance(timestamps, datetime):
        return get_hours_since_epoch(pd.Series([timestamps])).iloc[0]
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )


def get_datetime_from_hours_since_epoch(
    hours: Union[pd.Series, float, int],
) -> Union[pd.Series, datetime]:
    """
    Converts a given number of hours since the UNIX epoch into a UTC datetime.
    This is the inverse of the get_hours_since_epoch function.

    Args:
        hours: A float, int, or pandas Series representing the number of
               hours since 1970-01-01 00:00:00 UTC.

    Returns:
        A timezone-aware datetime object (or Series of objects) set to the UTC timezone.
    """
    if isinstance(hours, pd.Series):
        if len(hours) == 0:
            return pd.Series([], dtype="datetime64[ns, UTC]")

        # 1. Convert hours back to seconds, then to microseconds
        microseconds = hours * 3600 * 1_000_000

        # 2. Convert the float microseconds to integer representation
        integer_microseconds = microseconds.astype("int64")

        # 3. Convert integer microseconds since epoch to naive datetime objects
        #    The result represents UTC time but has no timezone info yet.
        naive_datetimes = pd.to_datetime(integer_microseconds, unit="us")

        # 4. Localize the naive datetimes to UTC to make them timezone-aware
        utc_datetimes = naive_datetimes.dt.tz_localize("UTC")

        return utc_datetimes

    elif isinstance(hours, (int, float)):
        # Handle a single number by wrapping it in a Series and extracting the result
        return get_datetime_from_hours_since_epoch(pd.Series([hours])).iloc[0]
    else:
        raise TypeError(
            "Invalid type for hours, only pd.Series, float, and int are supported."
        )
