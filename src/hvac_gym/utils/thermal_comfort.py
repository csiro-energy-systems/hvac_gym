# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.
import warnings
from datetime import datetime
from functools import lru_cache
from math import nan

import numpy as np
import pandas as pd
from joblib import Memory
from numpy._typing import NDArray
from pythermalcomfort import pmv_ppd
from pythermalcomfort.utilities import clo_dynamic, v_relative

memory = Memory("output/.func_cache", verbose=False)

default_clothing = {"summer": 0.5, "winter": 1.0, "shoulder": 0.75}


@lru_cache(maxsize=int(1e5), typed=False)
def interpolate_clothing(date: pd.Timestamp, summer_clo: float = 0.5, winter_clo: float = 1.0, north_hemisphere: bool = False) -> float:
    """Interpolates clothing value based on the hour of the year year using a cosine wave with period of a year.
    E.g. in the southern hemisphere, outputs summer_clo (0.5) on 1st Jan, winter_clo (1.0) on 1st July.  Vice versa for northern hemisphere.
    """
    date = pd.to_datetime(date)
    hour_of_year = date.dayofyear * 24 + date.hour
    hours_per_year = 365 * 24

    # phase shift summer to winter for southern hemisphere so max clothing level is in July
    if not north_hemisphere:
        hour_of_year = (hour_of_year + hours_per_year / 2) % hours_per_year

    return float(((np.cos(2 * np.pi * hour_of_year / hours_per_year) / 2 + 0.5) * (winter_clo - summer_clo)) + summer_clo)


@lru_cache(maxsize=int(1e7), typed=False)
def comfort_range(
    date: datetime, rh: float, pmv_lower: float, pmv_upper: float, temp_lower_limit: float = 10.0, temp_upper_limit: float = 40.0
) -> tuple[float | None, float | None]:
    """
    Get the comfort range for a given date, relative humidity and PMV range
    :param date: date to get comfort range for
    :param rh: relative humidity
    :param pmv_lower: lower PMV limit
    :param pmv_upper: upper PMV limit
    :param temp_lower_limit: lower limit of temperature search
    :param temp_upper_limit: upper limit of temperature search
    :return: tuple of lower and upper comfort temperatures. Values may be None if no temperature in the range satisfies PMV
    """
    with warnings.catch_warnings(action="ignore"):
        temp_upper = pmv_temperature_boundary(rh, date, pmv_upper, temp_lower_limit, temp_upper_limit)
        temp_lower = pmv_temperature_boundary(rh, date, pmv_lower, temp_lower_limit, temp_upper_limit)
    return temp_lower, temp_upper


""" Vectorised version of comfort_range for convenience (not speed), for calling with pandas/numpy arrays for date and rh inputs.
Other inputs remain as scalars. """
comfort_range_vectorised = np.vectorize(comfort_range, excluded=["pmv_lower", "pmv_upper", "offset", "temp_lower_limit", "temp_upper_limit"])


@lru_cache(maxsize=int(1e6), typed=False)
def pmv_temperature_boundary(
    rh: float,
    timestamp: datetime,
    pmv_target: float,
    temp_lower_limit: float = 10.0,
    temp_upper_limit: float = 40.0,
    offset: float = 0,
    skipna: bool = True,
    max_iters: int = 100,
    north_hemisphere: bool = False,
) -> float | None:
    """Search to determine the lower and upper ASHRAE 55 compliant temperatures

    For a given relative humidity, clo and met what are the temperatures closest to satisfying PMV +/- 0.5?

    Args:
        rh (float): space relative humidity
        timestamp (datetime): a datetime from which to calculate the season and interpolate clothing levels etc
        pmv_target (float): target predicted mean vote
        offset (float): mrt offset temperature
        temp_lower_limit (float): lower limit of temperature search
        temp_upper_limit (float): upper limit of temperature search
        skipna (bool): if True, return np.nan for missing values, otherwise attempts to calculate with NaNs.
        max_iters (int): maximum number of iterations to search for a temperature that satisfies pmv_target. Raises ArithmeticError if exceeded.

    Returns:
        float: temperature that satisfies pmv_target for given relative humidity and ASHRAE55 params or None if no temperature in the range
        satisfies pmv_target

    Raises:
        ArithmeticError: if max_iters is exceeded
    """
    # Starting temperature bounds
    temperature_lower = temp_lower_limit
    temperature_upper = temp_upper_limit

    if skipna:
        if pd.isna(rh):
            return nan

    season = get_season(timestamp)

    iters = 0
    while temperature_lower <= temperature_upper:
        iters += 1
        if iters > max_iters:
            raise ArithmeticError(
                f"Unable to find a temperature that satisfies PMV {pmv_target} for RH {rh} after max_iters > {max_iters} iterations"
            )

        mid = round((temperature_lower + temperature_upper) / 2, 2)
        tr = radiant_operative_temperature(mid, season, offset)[1]

        pmv, _ = calculate_pmv_ppd(mid, tr, rh, timestamp, north_hemisphere, skipna=skipna)

        if pmv < pmv_target:
            temperature_lower = mid
        elif pmv > pmv_target:
            temperature_upper = mid
        else:
            return mid
    return None


def radiant_operative_temperature(temperature: float, season: str, offset: float) -> tuple[float, float, float]:
    """Calculate radiant and operative_temperature given temperature, season and MRT offset

    Args:
        temperature (float): temperature of a space
        season (str): per the model's definition of seasons - summer, winter, shoulder
        offset (float): mean radiant temperature offset

    Returns:
        tuple: offset used, radiant temperature, and operative_temperature
    """
    if season == "Summer":
        temp_radiant = temperature + offset
        temp_operative = temperature + (0.5 * offset)
    elif season == "Winter":
        temp_radiant = temperature - offset
        temp_operative = temperature - (0.5 * offset)
    else:
        temp_radiant = temp_operative = temperature
    return offset, temp_radiant, temp_operative


@lru_cache(maxsize=int(1e6), typed=False)
def calculate_pmv_ppd(
    temperature: float,
    temp_radiant: float,
    relative_humidity: float,
    timestamp: datetime,
    north_hemisphere: bool = False,
    clothing: None | dict[str, float] = None,
    metabolic_rate: float = 1.1,
    air_velocity: float = 0.1,
    skipna: bool = False,
) -> tuple[float, float]:
    """Calculates the predicted mean vote (PMV) and predicted percent dissatisfied (PPD)

    TODO speed this up with a precalculated lookup table for default values and an interpolation function?

    Args:
        temperature (float): space temperature
        temp_radiant (float): radiant temperature
        relative_humidity (float): space relative humidity
        season (str): per the model's definition of seasons - summer, winter, shoulder
        clothing (dict[str, float]): clothing per season
        metabolic_rate (float): metabolic rate in kcal/kg/hour (or 'met' units).  1 met is equivalent to sitting quietly, 10 met is vigorous
        excercise.
        air_velocity (float): air velocity (m/s)
        skipna (bool): if True, return np.nan for missing values, otherwise attempts to calculate with NaNs.
    Returns:
        tuple: PMV and PPD
    """
    if skipna:
        if pd.isna([temperature, temp_radiant, relative_humidity, metabolic_rate, air_velocity]).any():
            return np.nan, np.nan

    # clothing per season
    clothing = default_clothing if clothing is None else clothing

    # Dynamic clothing
    # https://pythermalcomfort.readthedocs.io/en/latest/_modules/pythermalcomfort/utilities.html#clo_dynamic
    clo = interpolate_clothing(timestamp, clothing["summer"], clothing["winter"], north_hemisphere=north_hemisphere)
    cd = clo_dynamic(clo=clo, met=metabolic_rate, standard="ASHRAE")

    # Relative air speed
    # https://pythermalcomfort.readthedocs.io/en/latest/_modules/pythermalcomfort/utilities.html#v_relative
    vr = v_relative(v=air_velocity, met=metabolic_rate)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pmv: dict[str, float] = pmv_ppd(tdb=temperature, tr=temp_radiant, vr=vr, rh=relative_humidity, met=metabolic_rate, clo=cd, standard="ASHRAE")

    return pmv["pmv"], pmv["ppd"]


def get_season(timestamp: pd.Timestamp, north_hemisphere: bool = False) -> str:
    """Determine season (winter, summer shoulder) from timestamp."""
    timestamp = pd.to_datetime(timestamp)
    if north_hemisphere:
        if timestamp.month in [12, 1, 2]:
            return "winter"
        elif timestamp.month in [6, 7, 8]:
            return "summer"
        else:
            return "shoulder"
    else:
        if timestamp.month in [6, 7, 8]:
            return "winter"
        elif timestamp.month in [12, 1, 2]:
            return "summer"
        else:
            return "shoulder"


@memory.cache
def get_comfort_bounds(timestamps: pd.DatetimeIndex, oa_rh: pd.Series, pmv_lower: float, pmv_upper: float) -> tuple[NDArray[float], NDArray[float]]:
    """Get the comfort bounds for a given PMV range and relative humidity.
    Caches results speed to local dir (see `memory` var) for speedup on subsequent runs with identical inputs.
    :param timestamps: timestamps to get comfort bounds for
    :param oa_rh: outside air relative humidity
    :param pmv_lower: lower PMV limit
    :param pmv_upper: upper PMV limit
    """
    return comfort_range_vectorised(timestamps, oa_rh, pmv_lower, pmv_upper)  # type: ignore
