from dch.dch_interface import DCHBuilding
from dch.paths.dch_paths import SemPath, SemPaths
from pandas import DataFrame
from rdflib import BRICK

from hvac_gym.sites.model_config import HVACModel, HVACModelConf

""" Newcastle notes:
    - one office AHU per zone, 15 zones, 3 levels
    - 2 lab AHUs, 2 levels, 1 VAV per room

    1. Simple Whole-building average model:
    - Control average valve positions & fans speeds only
    - Model total chiller power as function of average valve position
        - (R^2=0.9, 200 days data, 50% split)
    - Model average zone temp as function of average valve positions, fan speeds and outside air temp
    (Ignore fan power for now - it's an order of magnitude lower than chiller power)
"""
# Custom SemPaths:
ambient_zone_temp = SemPath(
    name="ambient_zone_temp",
    path="Ambient_Zone_Temperature",
    description="Estimate of what the zone temperature would have been without any mechanical heating or cooling",
    validate_path=False,
)


def chiller_off_filter(filter_df: DataFrame, model_config: HVACModel) -> DataFrame:
    """Removes rows where the chiller isn't consuming significant power"""
    if SemPaths.chiller_elec_power.value in model_config.inputs:
        power_col = SemPaths.chiller_elec_power.value
        operating_power_threshold = (
            filter_df.between_time("01:00", "02:00")[power_col].quantile(0.9) * 1.2
        )
        not_operating_df = filter_df[filter_df[power_col] < operating_power_threshold]
        not_operating_df = not_operating_df.drop(columns=[power_col])
    return not_operating_df


""" An interim model that predicts the 'ambient_zone_temp' - ie what the zone temperature would have been without any mechanical heating or cooling
then we feed that prediction to the zone_temp model to modify it based on the chiller power, valves, fans etc """
ambient_zone_temp_model = HVACModel(
    target=SemPaths.zone_temp,
    inputs=[SemPaths.oa_temp, SemPaths.chiller_elec_power],
    output=ambient_zone_temp,
    horizon_mins=0,
    lags=[],
    scope=BRICK.Site,
    filters=[chiller_off_filter],
)

""" Simple model of average building zone temp as a function of hot/chilled water valves, fan speeds and outside air temp and the ambient_zone_temp
from the model above."""
zone_temp_model = HVACModel(
    target=SemPaths.zone_temp,
    inputs=[
        SemPaths.chiller_elec_power,
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_hw_valve_sp,
        SemPaths.ahu_sa_fan_speed,
        SemPaths.oa_temp,
        SemPaths.ahu_oa_damper,
    ],
    derived_inputs=[
        ambient_zone_temp
    ],  # predicted by the ambient_zone_temp_model above
    horizon_mins=0,
    lags=[],  # list(range(1, 7, 1)),
    lag_target=False,
    scope=BRICK.Site,
)

""" Models per-AHU electrical power, based on the AHU vs all chilled water valve positions and historical whole-site chiller power """
ahu_chws_elec_power_model = HVACModel(
    target=SemPaths.chiller_elec_power,
    inputs=[
        SemPaths.ahu_chw_valve_sp,
        SemPaths.oa_temp,
        SemPaths.ahu_sa_fan_speed,
        SemPaths.zone_temp,
    ],
    derived_inputs=[],
    horizon_mins=10,
    lags=[],  # list(range(1, 7, 1)),
    lag_target=False,
    scope=BRICK.Site,
)

model_conf = HVACModelConf(
    site=DCHBuilding("csiro", "newcastle", "Newcastle", tz="Australia/Sydney"),
    plot_data=False,
    sim_start_date="2023-10-01",  # Note: Chiller replaced in Sept 2023
    chiller_cop=2.0,
    resample_interval_mins=10,
    use_saved_models=False,
    refit_saved_models=False,
    save_refitted_models=True,
    tpot_max_time_mins=0,
    skopt_n_hyperparam_runs=0,
    setpoints=[
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_hw_valve_sp,
        SemPaths.ahu_sa_fan_speed_sp,
        SemPaths.ahu_oa_damper,
    ],
    # Warning: order is important here.  Need to specify shorter-horizon models first so that their predictions are used as inputs to
    # longer-horizon models.
    ahu_models=[
        ambient_zone_temp_model,
        zone_temp_model,
        ahu_chws_elec_power_model,
    ],
)
