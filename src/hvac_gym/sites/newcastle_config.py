from dch.dch_interface import DCHBuilding
from dch.paths.dch_paths import SemPaths
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

""" Models per-AHU electrical power, based on the AHU vs all chilled water valve positions and historical whole-site chiller power """
ahu_chws_elec_power = HVACModel(
    target=SemPaths.chiller_power,
    # inputs=[SemPaths.ahu_chw_valve_sp, SemPaths.oa_temp],  #  R2=0.704, RMSE=10.858
    # inputs=[SemPaths.ahu_chw_valve_sp, SemPaths.oa_temp, SemPaths.ahu_sa_fan_speed_sp],  # R2=0.737, RMSE=10.232
    # Lin:  R2=0.726, RMSE=8.538
    # RF:   R2=0., RMSE=
    # ELM:  R2=0.801, RMSE=7.266
    # ET:   R2=0., RMSE=
    inputs=[
        SemPaths.ahu_chw_valve_sp,
        SemPaths.oa_temp,
        SemPaths.ahu_sa_fan_speed,
        SemPaths.zone_temp,
    ],
    derived_inputs=[],
    horizon_mins=10,
    lags=list(range(1, 7, 1)),
    lag_target=False,
    scope=BRICK.Site,
)

""" Simple model of average building zone temp as a function of hot/chilled water valves, fan speeds and outside air temp """
sa_zone_temp = HVACModel(
    target=SemPaths.zone_temp,
    # inputs=[SemPaths.ahu_chw_valve_sp, SemPaths.ahu_hw_valve_sp, SemPaths.oa_temp],  # R2=0.474, RMSE=0.759
    # inputs=[SemPaths.ahu_chw_valve_sp, SemPaths.ahu_hw_valve_sp, SemPaths.ahu_sa_fan_speed_sp],  # R2=0.769, RMSE=0.504
    # inputs=[SemPaths.ahu_chw_valve_sp, SemPaths.ahu_hw_valve_sp, SemPaths.ahu_sa_fan_speed_sp, SemPaths.oa_temp],  # R2=0.780, RMSE=0.491
    # Lin:  R2=0.735, RMSE=0.584
    # RF:   R2=0., RMSE=0.
    # ET:   R2=0., RMSE=0.
    # ELM:  R2=0.798, RMSE=0.510
    inputs=[
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_hw_valve_sp,
        SemPaths.ahu_sa_fan_speed,
        SemPaths.oa_temp,
        SemPaths.ahu_oa_damper,
    ],
    derived_inputs=[],
    horizon_mins=0,
    lags=list(range(1, 7, 1)),
    lag_target=False,
    scope=BRICK.Site,
)

""" Models per-AHU fan power, based on the AHU's fan speed setpoint
Disabled because fan power measurements aren't available for all AHUs
"""
# fan_power = HVACModel(
#     target=SemPaths.ahu_sa_fan_power,
#     inputs=[SemPaths.ahu_sa_fan_speed_sp],
#     derived_inputs=[],
#     horizon_mins=10,
#     lags=[],
#     lag_target=False,
# )

sa_temp_model = HVACModel(
    target=SemPaths.ahu_sa_temp,
    inputs=[
        SemPaths.oa_temp,
        SemPaths.ahu_ra_temp,
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_hw_valve_sp,
        SemPaths.ahu_sa_fan_speed_sp,
        SemPaths.ahu_oa_damper,
    ],
    derived_inputs=[],  # [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=0,
    lags=list(range(0, 10, 1)),
    lag_target=False,
)

sa_rh_model = HVACModel(
    target=SemPaths.ahu_sa_rh,
    inputs=[
        SemPaths.ahu_ra_rh,
        SemPaths.ahu_oa_damper,
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_sa_fan_speed_sp,
        SemPaths.oa_rh,
    ],
    derived_inputs=[],  # =[ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=0,
    lags=list(range(0, 10, 2)),
    lag_target=False,
)
ra_temp_model = HVACModel(
    target=SemPaths.ahu_ra_temp,
    inputs=[SemPaths.ahu_sa_fan_speed_sp, SemPaths.ahu_sa_temp, SemPaths.oa_temp],
    derived_inputs=[],  # =[ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=10,
    lags=[
        *range(0, 6, 1),
        *range(6, 12, 2),
    ],  # RMSE=0.231, R2=0.978, SAT Importance = 0.01
    lag_target=False,
)
ra_rh_model = HVACModel(
    target=SemPaths.ahu_ra_rh,
    inputs=[SemPaths.ahu_sa_fan_speed_sp, SemPaths.ahu_sa_rh, SemPaths.oa_rh],
    derived_inputs=[],  # =[ModelPoint.minute_of_day, ModelPoint.day_of_week],
    horizon_mins=10,
    lags=list(range(0, 10, 2)),
    lag_target=True,
)

model_conf = HVACModelConf(
    site=DCHBuilding("csiro", "newcastle", "Newcastle", tz="Australia/Sydney"),
    plot_data=False,
    sim_start_date="2023-10-01",
    # Chiller replaced in Sept 2023
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
        sa_zone_temp,
        ahu_chws_elec_power,
    ],
    # , sa_temp_model, sa_rh_model, ra_temp_model, ra_rh_model],
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
