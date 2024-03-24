from dch.dch_interface import DCHBuilding
from paths.dch_paths import SemPaths

from hvac_gym.sites.model_config import HVACModel, HVACModelConf

sa_temp_model = HVACModel(
    target=SemPaths.ahu_sa_temp,
    inputs=[
        SemPaths.ahu_ra_temp,
        SemPaths.ahu_chw_valve_sp,
        SemPaths.ahu_hw_valve_sp,
        SemPaths.ahu_sa_fan_speed_sp,
        SemPaths.ahu_oa_damper,
        SemPaths.oa_temp,
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
    sim_start_date="2021-11-01",
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
    ahu_models=[sa_temp_model, sa_rh_model, ra_temp_model, ra_rh_model],
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
