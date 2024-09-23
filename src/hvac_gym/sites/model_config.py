# Created by wes148 at 1/07/2022
from pathlib import Path
from typing import Self

from dch.dch_interface import DCHBuilding
from dch.paths.dch_paths import SemPath
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, model_validator


class PathFilter(BaseModel):
    """Filter to apply to the dataframe before training the model"""

    def filter(self, filter_df: DataFrame, hvac_model_config: "HVACModelConf") -> DataFrame:
        """Apply the filter to the dataframe before training the model"""
        raise NotImplementedError

    def __hash__(self) -> int:
        """Hash the filter by its type and all values"""
        return hash((type(self),) + tuple(self.__dict__.values()))


class HVACModelConf(BaseModel):
    """Configuration for a single HVAC prediction/regression model"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # tell pydantic to allow dataframes etc

    # the target variable that the model predicts
    target: SemPath

    # inputs: list of ModelPoint to use as inputs to the model
    inputs: list[SemPath]

    # output: the output variable to predict. If None, the target variable is used.
    output: SemPath | None = None

    # horizon_mins: how far ahead to predict
    horizon_mins: int

    # derived_inputs: list of ModelPoint to derive from inputs
    derived_inputs: list[SemPath] = []

    # lags: list of lags to use for each input
    lags: list[int] = []

    # lag_target: whether to lag the target column as well as inputs
    lag_target: bool = False

    # adding temporal features to inputs
    add_temporal_features: bool = False

    # True will convert the gas usage from m3 to kW.
    convert_m3_kw: bool = False

    # filters: list of functions to apply to the dataframe before training the model
    filters: list[PathFilter] = []

    def __hash__(self) -> int:
        """Hashes all the attributes of the model, including list elements if theattribute is a list"""
        return hash(
            (
                self.target,
                tuple(self.inputs),
                self.output,
                self.horizon_mins,
                tuple(self.derived_inputs),
                tuple(self.lags),
                self.lag_target,
                tuple(self.filters),
            )
        )

    def __repr__(self) -> str:
        """Returns a string representation of the model"""
        return f"HVACModel(target={self.target}, output={self.output})"

    @model_validator(mode="after")
    def validate_target(self: Self) -> SemPath:
        """Normalise the target and output to a SemPath"""
        if self.output is None:
            self.output = self.target
        return self


class HVACSiteConf(BaseModel):
    """Configuration for a HVAC site, containing one or more HVACModels."""

    site: DCHBuilding

    plot_data: bool = False
    show_plots: bool = False

    # where to save/load output files
    out_dir: str = "output"

    # manually set a good starting point for the model. None to start at the beginning (after cleaning)
    # sim_start_date: datetime

    # Max allowable missing-data for rows and columns, above which the whole row/column will be dropped in preprocessing
    missing_threshold_rows: float = 0.75
    missing_threshold_cols: float = 0.5

    # Approximate chiller coefficient of performance (Watts_thermal / Watts_electric)
    chiller_cop: float = 2

    # Calorific value of the gas (kWh/m3)
    cv: float = 10

    # Boiler efficiency (%)
    boiler_efficiency: float = 0.75

    # Override automatic resampling of data to this interval in minutes
    resample_interval_mins: int = 10

    # Maximum number of consecutive missing values to interpolate over
    interpolation_limit: int = 6

    # Fraction of data to reserve for validation.  None will use every second day for validation.
    validate_frac: tuple[float, float] = (0.75, 1.0)

    # If true, skip all training, just used saved model pkl files.
    use_saved_models: bool = False

    # True will call model.fit() before validating. Necessary if you wish to preserve the modle hyperparams but the column count has changed etc.
    refit_saved_models: bool = False

    # Whether to overwrite existing model pkl file with refitted model.
    save_refitted_models: bool = True

    # TPOT automl settings.
    tpot_max_time_mins: int = 0  # 0 to disable.
    tpot_population_size: int = 20
    tpot_max_model_time_mins: int = 20
    tpot_early_stop: int = 3

    # skopt RF param tuning.  0 to disable
    n_hyperparam_runs: int = 0

    # Setpoints that agents' actions can change the value of. Not all models will use all setpoints.
    setpoints: list[SemPath]  # = [Path.chw_valve_sp, Path.hw_valve_sp, Path.fan_speed_sp, Path.oa_damper]

    # A list of data preprocessors, to apply to all training dataframes individually just before model fitting, for special cases.
    # Preprocessors must conform to the signature:
    #   `def preprocess(df: pd.DataFrame) -> dict[str, Any]`
    # Where key-value pairs must include "df: pd.DataFrame" and may optionally include "new_columns: list[str]"
    # Added columns will be added to every model's feature set.
    # preprocessors = [add_time_of_day_preprocessor, zero_fan_speed_when_off_preprocessor, relative_to_absolute_humidity_preprocessor]

    # Map of target_col to input_col_list for each model to build per AHU
    # To avoid leakage of real-world conditions in gym, model inputs should ONLY be other model targets/outputs, RL actions/setpoints,
    # or external sensors.
    # WARNING: order is important here.  Need to specify shorter-horizon models first so that their predictions are used as inputs to
    # longer-horizon models.
    ahu_models: list[HVACModelConf]

    def __str__(self: Self) -> str:
        return f"ModelConfig: {self.site}"

    #
    # ahu_models = {
    #     ModelPoint.sa_temp: {
    #         "inputs": [DCHPoints.ra_temp, ModelPoint.chw_valve_sp, ModelPoint.hw_valve_sp, ModelPoint.fan_speed_sp, ModelPoint.oa_damper,
    #         ModelPoint.oa_temp],
    #         "scope": BRICK.AHU,
    #         "derived_inputs": [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    #         "horizon_mins": 0,
    #         "lags": list(range(0, 10, 1)),
    #         "lag_target": False,
    #     },
    #     ModelPoint.sa_humidity: {
    #         "inputs": [ModelPoint.ra_humidity, ModelPoint.oa_damper, ModelPoint.chw_valve_sp, ModelPoint.fan_speed_sp, ModelPoint.oa_humidity],
    # ModelPoint.min_oa_damper
    #         "scope": BRICK.AHU,
    #         "derived_inputs": [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    #         "horizon_mins": 0,
    #         "lags": list(range(0, 10, 2)),
    #         "lag_target": False,
    #     },
    #     ModelPoint.ra_temp: {
    #         "inputs": [ModelPoint.fan_speed_sp, ModelPoint.sa_temp, ModelPoint.oa_temp],  # ModelPoint.ra_temp removed because models just learn
    #         to predict the previous value
    #         "scope": BRICK.AHU,
    #         "derived_inputs": [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    #         "horizon_mins": 10,
    #         # "lags": list(range(0, 60, 10)),             # RMSE=0.469, R2=0.909, SAT Importance = 0.01
    #         "lags": [*range(0, 6, 1), *range(6, 12, 2)],  # RMSE=0.231, R2=0.978, SAT Importance = 0.01
    #         "lag_target": False,                           # False: RMSE=0.705, R2=0.796, SAT Importance = 0.03 | True: RMSE=0.231, R2=0.978,
    #         SAT Importance = 0.01
    #         # TODO add feature?: (rat-sat) * fan_speed - this should be kinda proportional to the heat input from the AHU - large and positive
    #          when heating, large and negative when cooling.
    #     },
    #     ModelPoint.ra_humidity: {
    #         "inputs": [ModelPoint.fan_speed_sp, ModelPoint.sa_humidity, ModelPoint.oa_humidity],
    #         "scope": BRICK.AHU,
    #         "derived_inputs": [ModelPoint.minute_of_day, ModelPoint.day_of_week],
    #         "horizon_mins": 10,
    #         "lags": list(range(0, 10, 2)),
    #         "lag_target": True,
    #     },
    # }

    # power_models = {
    #     # Chilled water system power model predicts electrical (equivalent) power by predicting the chilled water loop's  using the proportion of
    #     the AHU's chilled water valve open % as a
    #     # fraction of all AHU chw valve %.
    #     # To simplify this modelling to suit real buildings with limited power measurements, this assumes that the flow/temp available to each
    #     AHU is identical,
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
    #
    # # All models together
    # predict_models = {**ahu_models, **power_models}


def save_config(conf: HVACSiteConf, path: Path) -> None:
    """Save the configuration to a JSON file"""
    with open(path, "w") as f:
        f.write(conf.model_dump_json(indent=4, round_trip=True))


def load_config(path: Path) -> HVACSiteConf:
    """Load the configuration from a JSON file"""
    with open(path) as f:
        result = HVACSiteConf.model_validate_json(f.read())
    return result  # type: ignore
