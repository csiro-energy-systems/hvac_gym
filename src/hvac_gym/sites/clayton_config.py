# The Software is copyright (c) CSIRO ABN 41 687 119 230
from dch.dch_interface import DCHBuilding
from dch.paths.sem_paths import (
    ahu_chw_valve_sp,
    ahu_enable_status,
    ahu_hw_valve_sp,
    ahu_oa_damper,
    ahu_room_temp,
    ahu_sa_temp,
    ahu_zone_temp,
    oa_temp,
)

from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf

""" Clayton notes:
    - fixed-speed AHU fans (just have an on-off status point),
    - no zone or supply air humidity sensors
    - no zone RH sensors, so no RH models
"""

sa_temp_model = HVACModelConf(
    target=ahu_sa_temp,
    inputs=[
        ahu_zone_temp | ahu_room_temp,
        oa_temp,
        ahu_chw_valve_sp,
        ahu_hw_valve_sp,
        ahu_enable_status,
        ahu_oa_damper,
    ],
    derived_inputs=[],  # [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=0,
    lags=list(range(0, 10, 1)),
    lag_target=False,
)
zone_temp_model = HVACModelConf(
    target=ahu_zone_temp,
    inputs=[ahu_enable_status, ahu_sa_temp, oa_temp],
    derived_inputs=[],  # =[ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=10,
    lags=[
        *range(0, 6, 1),
        *range(6, 12, 2),
    ],  # RMSE=0.231, R2=0.978, SAT Importance = 0.01
    lag_target=False,
)

model_conf = HVACSiteConf(
    site=DCHBuilding("csiro", "clayton", "Building307", tz="Australia/Melbourne"),
    plot_data=False,
    sim_start_date=None,
    chiller_cop=2.0,
    resample_interval_mins=10,
    use_saved_models=False,
    refit_saved_models=False,
    save_refitted_models=True,
    tpot_max_time_mins=0,
    skopt_n_hyperparam_runs=0,
    setpoints=[
        ahu_chw_valve_sp,
        ahu_hw_valve_sp,
        ahu_enable_status,
        ahu_oa_damper,
    ],
    ahu_models=[sa_temp_model, zone_temp_model],
    power_models=[],
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
