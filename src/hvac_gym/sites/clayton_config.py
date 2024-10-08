from typing import Literal

from dch.dch_interface import DCHBuilding
from dch.paths.dch_paths import SemPath

# from dch.paths.sem_paths import (
#    ahu_chw_valve_sp,
#    ahu_hw_valve_sp,
#    ahu_oa_damper,
#    ahu_room_temp,
#    oa_temp,
# )
from overrides import overrides
from pandas import DataFrame
from pydantic import BaseModel

from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf, PathFilter

# FIXME this must use a general path to the gas meter data.
# gas_meter = SemPath(
# name="gas_meter",
# path=[
# "Building_Gas_Meter hasPoint Usage_Sensor[name_path=='GasMt|GM006']"
# ],
# override=True)
# FIXME GasMt|GM006 was removed in DCH 2.0. this GasMtPH caused significant drop in gas usage model metrics.
gas_meter = SemPath(
    name="gas_meter",
    path=["Building_Gas_Meter hasPoint Usage_Sensor[name_path=='GasMt|Clayton.ClaytonB307.Misc_LvlB.Misc_Bldg307_LvlB_GasMtPH']"],
    override=True,
)

chiller_elec_power = SemPath(
    name="chiller_elec_power",
    path=["Chiller isMeteredBy Electrical_Meter hasPoint Electric_Power_Sensor[unit=='unit:KiloW' and electricalPhases=='ABC']"],
    override=True,
)

# FIXME currently override=False returns error
ahu_chw_valve_sp = SemPath(
    name="ahu_chw_valve_sp",
    path="AHU hasPart Chilled_Water_Coil feeds Chilled_Water_Valve hasPoint Valve_Position_Sensor",
    normalise_range=[0.0, 1.0],
    override=True,
)
ahu_hw_valve_sp = SemPath(
    name="ahu_hw_valve_sp",
    path="AHU hasPart Hot_Water_Coil feeds Hot_Water_Valve hasPoint Valve_Position_Sensor",
    normalise_range=[0.0, 1.0],
    override=True,
)
ahu_oa_damper = SemPath(
    name="ahu_oa_damper", path=["AHU hasPart Outside_Damper hasPoint Damper_Position_Sensor"], normalise_range=[0.0, 1.0], override=True
)
ahu_room_temp = SemPath(name="ahu_room_temp", path="AHU feeds HVAC_Zone hasPart Room hasPoint Air_Temperature_Sensor", override=True)
oa_temp = SemPath(
    name="oa_temp",
    path=["Equipment hasPoint Outside_Air_Temperature_Sensor", "Outside_Air_Temperature_Sensor"],
    desc="All outside air temperature sensors",
    override=True,
)
ahu_sa_fan_speed = SemPath(name="ahu_sa_fan_speed", path="AHU hasPart Supply_Fan hasPoint Speed_Sensor", override=True)


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
# Custom SemPath:
ambient_zone_temp = SemPath(
    name="ambient_zone_temp",
    path="Ambient_Zone_Temperature",
    description="Estimate of what the zone temperature would have been without any mechanical heating or cooling",
    validate_path=False,
    override=True,
)


class ChillerOffFilter(PathFilter, BaseModel):
    """Removes rows where the boiler isn't consuming significant power"""

    type: Literal["ChillerOffFilter"] = "ChillerOffFilter"

    @overrides
    def filter(self, filter_df: DataFrame, hvac_model_config: HVACModelConf) -> DataFrame:
        """Removes rows where the boiler isn't consuming significant power"""
        if chiller_elec_power in hvac_model_config.inputs:
            power_col = chiller_elec_power.name
            operating_power_threshold = filter_df.between_time("01:00", "02:00")[power_col].quantile(0.9) * 1.2
            not_operating_df = filter_df[filter_df[power_col] < operating_power_threshold]
            not_operating_df = not_operating_df.drop(columns=[power_col])
            return not_operating_df
        else:
            return filter_df


""" An interim model that predicts the 'ambient_zone_temp' - ie what the zone temperature would have been without any mechanical heating or cooling
then we feed that prediction to the zone_temp model to modify it based on the boiler power, valves, fans etc """
ambient_zone_temp_model = HVACModelConf(
    target=ahu_room_temp,
    inputs=[
        oa_temp,
        chiller_elec_power,
    ],
    output=ambient_zone_temp,
    horizon_mins=0,
    lags=[],
    filters=[ChillerOffFilter()],
    add_temporal_features=False,
)

""" Simple model of average building zone temp as a function of hot/chilled water valves, fan speeds and outside air temp and the ambient_zone_temp
from the model above."""
zone_temp_model = HVACModelConf(
    target=ahu_room_temp,
    inputs=[
        chiller_elec_power,
        gas_meter,
        ahu_chw_valve_sp,
        ahu_hw_valve_sp,
        # ahu_sa_fan_speed,
        # oa_temp,
        ahu_oa_damper,
    ],
    # predicted by the ambient_zone_temp_model above
    derived_inputs=[ambient_zone_temp],
    horizon_mins=0,
    lags=[],  # list(range(1, 7, 1)),
    add_temporal_features=False,
    lag_target=False,
)

""" Models per-AHU electrical power, based on the AHU vs all chilled water valve positions and historical whole-site chiller power """
ahu_chws_elec_power_model = HVACModelConf(
    target=chiller_elec_power,
    inputs=[
        ahu_chw_valve_sp,
        oa_temp,
        #  ahu_sa_fan_speed,
        ahu_room_temp,
    ],
    derived_inputs=[],
    horizon_mins=10,
    lags=[],  # list(range(1, 7, 1)),
    add_temporal_features=False,
    lag_target=False,
)

""" Models per-AHU heating power, based on the AHU vs all chilled water valve positions and historical whole-site gas usage """
ahu_hws_power_model = HVACModelConf(
    target=gas_meter,
    inputs=[
        ahu_hw_valve_sp,
        oa_temp,
        #    ahu_sa_fan_speed,
        ahu_room_temp,
    ],
    derived_inputs=[],
    horizon_mins=10,
    lags=[],  # list(range(1, 7, 1)),
    add_temporal_features=False,
    convert_m3_kw=True,
    lag_target=False,
)

model_conf = HVACSiteConf(
    site=DCHBuilding("csiro", "Clayton", "building307", tz="Australia/Melbourne"),
    plot_data=False,
    sim_start_date="2023-01-01",
    chiller_cop=2.0,
    cv=9.0,
    boiler_efficiency=0.75,
    resample_interval_mins=10,
    use_saved_models=False,
    refit_saved_models=False,
    save_refitted_models=True,
    tpot_max_time_mins=0,
    skopt_n_hyperparam_runs=0,
    out_dir="output",
    setpoints=[
        ahu_chw_valve_sp,
        ahu_hw_valve_sp,
        ahu_oa_damper,
        # ahu_sa_fan_speed_sp,
    ],
    # Warning: order is important here.  Need to specify shorter-horizon models first so that their predictions are used as inputs to
    # longer-horizon models.
    ahu_models=[ambient_zone_temp_model, zone_temp_model, ahu_chws_elec_power_model, ahu_hws_power_model],
)

# power_models = {
#     # Chilled water system power model predicts electrical (equivalent) power by predicting the chilled water loop's  using the proportion of
#     # the AHU's chilled water valve open % as a
#     # fraction of all AHU chw valve %.
#     # To simplify this modelling to suit real buildings with limited power measurements, this assumes that the flow/temp available to each AHU
#     # is identical,
#     # and that flow is proportional to valve open %.
#     ModelPoint.ahu_chws_electrical_power: {
#         'inputs': [ModelPoint.chw_valve_sp, ModelPoint.oa_temp, ],
#         'scope': BRICK.Building,
#         'derived_inputs': [],
#         'horizon_mins': 10,
#         'lags': [],
#         "lag_target": False,
#     },
#     # Fan power model uses a static power-law to predict fan power as a function of fan seed (setpoint)
#     ModelPoint.fan_power: {
#         'inputs': [ModelPoint.fan_speed_sp],
#         'scope': BRICK.AHU,
#         'derived_inputs': [],
#         'horizon_mins': 10,
#         'lags': [],
#         "lag_target": False,
#     },
# }

# # All models together
# predict_models = {**ahu_models, **power_models}
