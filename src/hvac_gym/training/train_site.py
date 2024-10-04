import warnings
from datetime import datetime, timedelta
from pathlib import Path

import dill
import matplotlib

matplotlib.use("Agg")

import numpy as np
import optuna
import pandas as pd
import plotly.io as pio
from dch.dch_interface import DCHBuilding, DCHInterface
from dch.paths.dch_paths import SemPath
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp
from dch.utils.data_utils import resample_and_join_streams
from dch.utils.dch_model_utils import flatten
from dch.utils.init_utils import cd_project_root
from dotenv import load_dotenv
from joblib import Memory
from optuna import Trial
from pandas import DataFrame, Series
from plotly import express as px
from plotly.graph_objs import Figure
from pydantic import BaseModel, ConfigDict
from skelm import ELMRegressor
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from hvac_gym.config.log_config import get_logger
from hvac_gym.sites import clayton_config
from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf
from hvac_gym.training.train_utils import split_alternate_days
from hvac_gym.vis.vis_tools import figs_to_html

pio.templates.default = "plotly_dark"

logger = get_logger()
cd_project_root()

draw_plots = False  # enables/disables plotting during training
show_plots = False  # enables/disables opening plots when draw_plots==True

pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 10, "display.max_columns", 20, "display.width", 200, "display.max_colwidth", 30)

func_cache_dir = Path("output/.func_cache")
memory = Memory(func_cache_dir, verbose=1)
memory.reduce_size(bytes_limit="1G", age_limit=timedelta(days=365))
logger.info(f"Using function cache dir at: {func_cache_dir}")


class TrainingData(BaseModel):
    """A set of data and metadata for a set of related point_id s"""

    # tell pydantic to allow dataframes etc
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: DataFrame
    point_id: DataFrame
    sample_rate: float


class TrainingSet(BaseModel):
    """A set of multiple `TrainingData` input and target instances for a single physical model"""

    # tell pydantic to allow dataframes etc
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: TrainingData
    inputs: list[TrainingData]


class TrainSite:
    """For a given site, iterates through all model definitions, gets/preprocesses data for all available AHUs and trains predictive ML models on
    each.
    """

    def __init__(self) -> None:
        """Create a new TrainSite instance"""
        load_dotenv()
        self.dch = DCHInterface()

    def check_streams(self, model_conf: HVACModelConf, building: DCHBuilding) -> dict[str, dict[SemPath, DataFrame]]:
        """Check that necessary point_id exist
        :param model_conf: the model configuration
        :param building: the building to check point_id for
        :return:
        """
        target_point = model_conf.target
        input_points = model_conf.inputs

        target_stream = self.dch.path_search(building, target_point, long_format=True)
        input_streams = {point: self.dch.path_search(building, point, long_format=True) for point in input_points}

        missing_streams = ([point for point, point_id in input_streams.items() if point_id.empty]) + ([target_point] if target_stream.empty else [])
        if len(missing_streams) > 0:
            logger.error(f"Missing target or input point_id for {building}. Missing inputs: {missing_streams}")

        return {
            "target_streams": {target_point: target_stream},
            "input_streams": input_streams,
        }

    def transform_data(self, df: DataFrame) -> DataFrame:
        """Apply cyclic transformations to the dataframe and return it."""
        for period, cols in [
            ("month", "Month"),
            ("hour", "Hour"),
            ("weekday", "Weekday"),
        ]:
            sin_col, cos_col = self.cyclic_transform(df.index.to_series().dt.__getattribute__(period))
            df[f"Sine Transformed {cols} (n/a)"] = sin_col
            df[f"Cosine Transformed {cols} (n/a)"] = cos_col

        return df

    def cyclic_transform(self, column: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Convert cyclic features using sine and cosine transformations."""
        column = column + 1
        max_value = column.max()
        sin_values = np.sin(2 * np.pi * column / (max_value + 0.00001))
        cos_values = np.cos(2 * np.pi * column / (max_value + 0.00001))
        return sin_values, cos_values

    def m3_to_kw(self, df: DataFrame, target: SemPath, cv: float, efficiency: float) -> DataFrame:
        """
        Convert gas usage from m³ to kW.

        :param df (DataFrame): data.
        :param sample_rate (float): sample rate in seconds.
        :param cv (float): calorific value in kWh/m³.
        :param efficiency (float): efficiency of the boiler.
        :return df (DataFrame): dataframe with gas meter column converted to power in kW.
        """
        gas_usage = df[target]
        time_hours = (df.attrs["sample_rate_mins"]) / 60
        energy_kwh = gas_usage * cv * efficiency
        power_kw = energy_kwh / time_hours
        df[target] = power_kw
        return df

    def preprocess_data(
        self,
        data_set: TrainingSet,
        site_conf: HVACSiteConf,
        model_conf: HVACModelConf,
        point_id: dict[str, DataFrame],
    ) -> DataFrame:
        """Preprocesses the training data for each AHU and model
        :param point_id:
        """
        target = data_set.target
        inputs = data_set.inputs

        # concat input all point_id into a dataframe
        building_input_streams = pd.concat([i.point_id for i in inputs])
        building_cols = list(flatten(building_input_streams["data_column_name"]))
        building_df, sample_rate = resample_and_join_streams([i.data[[c for c in building_cols if c in i.data.columns]] for i in inputs])

        self.sample_rate = sample_rate

        # take median of each group of building data types (e.g. there's often several OAT sensors)
        # use SemPath instances for new aggregated column names
        building_stream_types = building_input_streams.groupby("type_path").aggregate({"data_column_name": lambda x: list(x)})

        for stream_type in building_stream_types.index:
            cols = [c for c in building_stream_types.loc[stream_type, "data_column_name"] if c in building_df.columns]
            sempath = building_input_streams.query("type_path == @stream_type")["sem_path"].unique()[0]
            building_df[sempath] = building_df[cols].median(axis=1, skipna=True)
            # building_df[f"{stream_type}_total"] = building_df[cols].sum(axis=1, skipna=True)
            building_df = building_df.drop(columns=cols)

        input_cols = list(building_df.columns)
        target_col = model_conf.target

        # adding temporal features
        if model_conf.add_temporal_features:
            building_df = self.transform_data(building_df.copy())

        input_cols = list(building_df.columns)
        target_col = model_conf.target

        # append the median of the target data to the building data
        target_cols = list(target.point_id["data_column_name"])
        target_df = target.data[target_cols]
        target_df = pd.DataFrame(target_df.median(axis=1), columns=[target_col])

        building_df = (
            target_df.resample(f"{sample_rate}min").median().join(building_df).interpolate(limit=6, limit_direction="forward", method="linear")
        )

        # add lagged columns for each input.
        # don't add lag to the temporal features.
        exclude_cols = [col for col in building_df.columns if "Transformed" in col]

        lagged_cols = []
        for col in input_cols:
            if col in exclude_cols:
                continue
            for lag in model_conf.lags:
                lag_col = f"{col}_{int(lag * sample_rate)}min"
                building_df[lag_col] = building_df[col].shift(lag)
                lagged_cols.append(lag_col)

        # add metadata to the dataframe
        building_df.attrs["lagged_cols"] = lagged_cols
        building_df.attrs["input_cols"] = input_cols
        building_df.attrs["target_col"] = target_col
        building_df.attrs["sample_rate_mins"] = target.sample_rate
        building_df.attrs["point_id"] = point_id

        # shift (non-target) inputs down by the horizon so they are used to forecast future targets
        horizon_mins = model_conf.horizon_mins
        horizon_rows = int(horizon_mins / target.sample_rate)
        input_cols = [c for c in building_df.columns if c != target_col]
        building_df[input_cols] = building_df[input_cols].shift(horizon_rows)

        return building_df

    def run(self, site_config: HVACSiteConf, start: datetime, end: datetime) -> None:
        """Main entrypoint to acquire data, train models and run gym-like simulation for a site"""
        models: dict[HVACModelConf, Pipeline] = {}
        model_dfs = []
        predictions: list[Series] = []
        for model_conf in site_config.ahu_models:
            logger.info(f"Training model {model_conf.output} model for {site_config.site}")
            building = site_config.site

            # Gather and preprocess data
            point_id = self.check_streams(model_conf, building)
            point_id = self.validate_streams(point_id)

            data = get_data(point_id, building, start, end)

            building_df = self.preprocess_data(data, site_config, model_conf, point_id)

            # FIXME it tries to convert gas meter data to power (in kW).
            # the function itself needs to be more general.
            # also conversion constants are just based on assumptions.
            if hasattr(model_conf, "convert_m3_kw") and model_conf.convert_m3_kw:
                building_df = self.m3_to_kw(building_df.copy(), model_conf.output, cv=site_config.cv, efficiency=site_config.boiler_efficiency)

            model, preds = train_site_model(building_df, site_config.site, model_conf, predictions)
            predictions.append(preds)
            models[model_conf] = model

            # Save the preprocessed DF from each model.  Inputs are shifted so ready for prediction.
            model_dfs.append(building_df)

        sim_df = self.save_models_and_data(models, model_dfs, site_config)

        # Step the models through a simulation period. Predict with each in the specified order then update the dataframe with predictions
        self.simulate(sim_df, models, site_config, sim_steps=200)

        # or profile with cProfile
        # import cProfile
        # cProfile.runctx('self.simulate(sim_df, models, site_config)', globals(), locals(), filename='simulate.prof')

    def simulate(self, sim_df: DataFrame, models: dict[HVACModelConf, RegressorMixin], site_config: HVACSiteConf, sim_steps: int = 500) -> None:
        """
        Simple gym-like simulation with the models and combined dataframe. Mainly used for manual verification that trained models interact and
        respond to actions as expected.
        :param sim_df: the dataframe to simulate
        :param models: a dict of target:train-model pairs
        :param site_config: the site configuration
        :param sim_steps: the number of time steps to simulate, or None to simulate using whole dataframe
        """
        for idx, time in enumerate(tqdm(sim_df.index[:sim_steps], desc="Simulating", unit="steps")):
            all_inputs_no_lags = []

            for model_conf in site_config.ahu_models:
                model = models[model_conf]

                inputs = model.feature_names_in_
                output = model_conf.output
                model_df = sim_df[inputs]

                inputs_no_lags = model.attrs["input_cols"] + [str(i) for i in model_conf.derived_inputs]
                all_inputs_no_lags.extend(inputs_no_lags)

                # set chilled water valve to square wave, cycling every N steps
                chwv_col = str(ahu_chw_valve_sp.name)
                hwv_col = str(ahu_hw_valve_sp.name)
                cycle_steps = 12 * 2

                chwv_sp = 100 if idx % cycle_steps < cycle_steps / 2 else 0
                sim_df.loc[time, chwv_col] = chwv_sp

                hwv_sp = 100 if idx % cycle_steps > cycle_steps / 2 else 0
                sim_df.loc[time, hwv_col] = hwv_sp

                # TODO set lags here also

                # get the model's prediction for the next timestep.
                # note: inputs/lags have already been down-shifted, so we can just apply prediction to the inputs at the timestamp directly.
                # this prediction will _apply_ to timestamp + horizon_mins for the model though
                predict_df = model_df.loc[[time]]

                # Run the prediction and update the building_df with result
                prediction = model.predict(predict_df)
                predict_time = time + timedelta(minutes=model_conf.horizon_mins)
                sim_df.loc[predict_time, str(output)] = prediction

        # plot the simulation results with lines
        title = f"Simulation results for {site_config.site}"
        targets = [str(m.target) for m in site_config.ahu_models]
        actuals = [c for c in sim_df.columns if c.endswith("_actual")]
        p = sim_df[targets + [str(i) for i in all_inputs_no_lags] + list(actuals)].melt(ignore_index=False)
        fig = px.line(p, x=p.index, y="value", color="variable", title=title)
        figs_to_html([fig], f"output/{title}", show=show_plots, verbose=1)

    def validate_streams(self, point_id: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Validate the point_id and check for common AHUs between target and input point_id"""
        target_streams = list(point_id["target_streams"].values())[0]
        input_streams = point_id["input_streams"]

        def append_ahu_name(point_id: DataFrame) -> DataFrame:
            if "type_path" in point_id.columns:
                point_id["ahu_point"] = point_id["type_path"].str.split("|").apply(lambda x: x[0] == "AHU")
            if "name_path" in point_id.columns:
                point_id["ahu_name"] = point_id.apply(
                    lambda x: x["name_path"].split("|")[0] if x["ahu_point"] else None,
                    axis=1,
                )
            return point_id

        # Get list of AHUs that the target point_id apply to, if any
        target_ahus: set[str] = set()
        append_ahu_name(target_streams)
        if "ahu_name" in target_streams.columns:
            target_ahus = set(target_streams["ahu_name"].to_list())
            logger.info(f"Found target AHUs: {target_ahus}")

        # Get list of AHUs that the input point_id apply to, if any
        for point in input_streams.keys():
            append_ahu_name(input_streams[point])

        input_ahu_dict = {
            point: (set(input_streams[point]["ahu_name"]) if "ahu_name" in input_streams[point].columns else None) for point in input_streams.keys()
        }

        input_ahus = [set(ahus) for ahus in input_ahu_dict.values() if ahus is not None]

        # check if there are common AHUs between target and all the input point_id
        # TODO implement per-AHU modelling where possible. This isn't used yet.
        _common_ahus = set(target_ahus).intersection(*input_ahus)

        return point_id

    def save_models_and_data(self, models: dict[HVACModelConf, Pipeline], model_dfs: list[DataFrame], site_config: HVACSiteConf) -> DataFrame:
        """
        Combines dataframes from all models into a single DF that can be used for simulation, and saves it and the models to disk.
        :param models: the trained models
        :param model_dfs: the dataframes for each model
        :param site_config: the site configuration
        :return: the combined dataframe
        """
        # save models to disk
        for model_conf, model in models.items():
            with open(f"{site_config.out_dir}/{site_config.site}_{model_conf.output}_model.pkl", "wb") as f:
                dill.dump(model, f)

        # combine DFs from all models
        sim_df = pd.concat(model_dfs, axis="columns")

        # make targets the first columns
        sim_df = sim_df[
            sorted(
                sim_df.columns,
                key=lambda x: x not in [m.target for m in site_config.ahu_models],
            )
        ]

        # remove duplicate named columns
        sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]
        sim_df = sim_df.dropna(how="any")
        sim_df.columns = [str(c) for c in sim_df.columns]

        # extract target, actuals for plotting later
        targets = [str(m.target) for m in site_config.ahu_models]
        actuals = sim_df[targets].copy()
        actuals.columns = [f"{c}_actual" for c in actuals.columns]

        # Align indices before joining
        actuals = actuals.reindex(sim_df.index)
        sim_df = actuals.join(sim_df)  # add actuals for debugging
        sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]

        sim_df.to_parquet(f"output/{site_config.site}_sim_df.parquet")
        return sim_df


def plot_df(df: DataFrame, title: str, out_dir: Path | None = None) -> Figure:
    """Plot a dataframe using plotly"""
    fig = px.line(df, height=1600)
    fig.update_layout(title=title)
    if out_dir:
        figs_to_html([fig], out_dir / f"{title}.html", show=show_plots)
    return fig


def split_df(train_test_df: DataFrame, target_cols: list[str]) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Split a dataframe into training and testing sets using alternate day strategy"""
    train_test_df.columns = [str(c.name) if isinstance(c, SemPath) else str(c) for c in train_test_df.columns]
    train_test_df = train_test_df.sort_index().dropna()
    train, test = split_alternate_days(train_test_df, n_sets=2)

    train_x, train_y = train.drop(columns=target_cols), train[target_cols]
    test_x, test_y = test.drop(columns=target_cols), test[target_cols]
    return train_x, train_y, test_x, test_y


# @memory.cache()
def get_data(
    point_id: dict[str, DataFrame],
    building: DCHBuilding,
    start: datetime,
    end: datetime,
) -> TrainingSet:
    """
    Gets a dictionary of dataframes for the target and input point_id
    Each
    :param point_id: dictionary of target and input point_id to get data for
    :param building: the building to get data for
    :param start: data start date
    :param end: data end date
    :return: A TrainingSet object with data and metadata for all point_id
    """
    target_path = list(point_id["target_streams"].keys())[0]
    target_streams = point_id["target_streams"][target_path]
    target_streams = target_streams.loc[np.logical_not(target_streams["point_id"].isna())]

    # FIXME some point_id s are nan when querying for oa_temp
    for point in point_id["input_streams"].keys():
        if "point_id" in point_id["input_streams"][point].columns:
            point_id["input_streams"][point] = point_id["input_streams"][point].loc[point_id["input_streams"][point]["point_id"].notna()]

    dch = DCHInterface()
    target = dch.get_joined_point_data(
        target_streams["point_id"].explode().unique().tolist(),
        start=start,
        end=end,
    )

    target["data"], target_streams = dch.building_connector.rename_data_columns(target["data"], target_streams, ["subtype_path"])

    # add column containing SemPath instance for later use
    target_streams["sem_path"] = [str(target_path)] * len(target_streams)
    # drop target_strems rows for which we have no columns
    target_streams = target_streams.query("data_column_name in @target['data'].columns")

    target_data = TrainingData(
        data=target["data"],
        point_id=target_streams,
        sample_rate=target["sample_rate"],
    )

    inputs = [
        dch.get_joined_point_data(
            point_id["input_streams"][point]["point_id"].explode().unique().tolist(),
            start=start,
            end=end,
        )
        for point in point_id["input_streams"].keys()
        if "point_id" in point_id["input_streams"][point]
    ]

    # rename data columns using brick classes, and add point_id to inputs dict
    for i, point in tqdm(enumerate(point_id["input_streams"].keys()), desc="Getting input data", total=len(inputs)):
        inputs[i]["data"], inputs[i]["point_id"] = dch.building_connector.rename_data_columns(
            inputs[i]["data"], point_id["input_streams"][point], ["subtype_path"]
        )

        # add column containing SemPath instances for later use
        inputs[i]["point_id"]["sem_path"] = [str(point)] * len(inputs[i]["point_id"])

        # remove any stream rows that we don't have data for
        inputs[i]["point_id"] = inputs[i]["point_id"].query("data_column_name in @inputs[@i]['data'].columns")

    input_data = [TrainingData(data=i["data"], point_id=i["point_id"], sample_rate=i["sample_rate"]) for i in inputs]

    training_set = TrainingSet(target=target_data, inputs=input_data)

    return training_set


def elm_optuna_param_search(x_train: pd.DataFrame, y_train: pd.DataFrame, n_trials: int = 500) -> RegressorMixin:
    """
    Performs an optuna search for the best hyperparameters for an ELMRegressor model, trying to maximise cross-validation score on the
    provided training set.
    :param x_train: the training set features
    :param y_train: the training set target
    :param n_trials: the number of optuna trials to perform
    :return: the best model (unfitted, with best found params) and the optuna optimization history plot
    """

    def objective(trial: Trial) -> float:
        """Optuna search for the ELMRegressor hyperparams"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor_obj = ELMRegressor(
                n_neurons=trial.suggest_int("n_neurons", 10, 1000, log=False),
                ufunc=trial.suggest_categorical("ufunc", ["tanh", "sigm", "relu", "lin"]),
                alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                include_original_features=trial.suggest_categorical("include_original_features", [True, False]),
                density=trial.suggest_float("density", 1e-3, 1, step=0.1),
                pairwise_metric=trial.suggest_categorical("pairwise_metric", ["euclidean", "cityblock", "cosine", None]),
            )

        try:
            score = cross_val_score(regressor_obj, x_train, y_train, n_jobs=-1, cv=3)
            accuracy = float(score.mean())
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            return float("-inf")
        return accuracy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", trial.value)
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    model = ELMRegressor(**trial.params)
    return model


# @memory.cache()
def train_site_model(
    train_test_df: DataFrame,
    site: DCHBuilding,
    model_conf: HVACModelConf,
    predictions: list[Series],
) -> RegressorMixin:
    """
    Train a model for a single site and model configuration
    To allow for chained models (output of one is input to next), we return the (train+test) predictions for each model, accumulate into a list
    and pass them back as `predictions` here for subsequent models. These are appended to any HVACModels which have matching `derived_inputs`
    properties.
    :param train_test_df:
    :param site:
    :param model_conf:
    :param predictions: accumulated predictions from prior models.
    :return:
    """
    target_cols = [model_conf.target.name]
    _input_cols = train_test_df.attrs["input_cols"]
    lagged_cols = train_test_df.attrs["lagged_cols"]
    _sample_rate = train_test_df.attrs["sample_rate_mins"]
    _target_col = train_test_df.attrs["target_col"]
    _target_streams = train_test_df.attrs["point_id"]["target_streams"]

    train_test_df = train_test_df.resample(f"{_sample_rate}min").median().copy()

    # add any derived inputs produced by previous training runs
    if model_conf.derived_inputs:
        predictions_df = DataFrame(pd.concat(predictions))
        for derived_input in model_conf.derived_inputs:
            if derived_input in predictions_df.columns:
                train_test_df[derived_input] = predictions_df[derived_input]

    unfiltered_df = train_test_df.copy()

    # apply dataset filters
    for fltr in model_conf.filters:
        train_test_df = fltr.filter(train_test_df, model_conf)

    # split the data into training and testing sets
    train_x, train_y, test_x, test_y = split_df(train_test_df, target_cols)
    train_x_unfilt, train_y_unfilt, test_x_unfilt, test_y_unfilt = split_df(unfiltered_df, target_cols)

    # linear models
    # model = sklearn.linear_model.LinearRegression()
    model = ElasticNetCV(n_jobs=-1)  # very fast fitting

    # model = sklearn.linear_model.LassoCV(n_jobs=-1)

    # nonlinear models
    # model = ELMRegressor(**{'n_neurons': 119,'ufunc': 'sigm','alpha': 0.011,'include_original_features': True,'density': 0.7,
    # 'pairwise_metric': 'euclidean'})
    # model = elm_optuna_param_search(train_x, train_y.to_numpy().ravel(), n_trials=500)
    # model = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, verbose=0, n_estimators=30)
    # model = ExtraTreesRegressor(n_jobs=-1, n_estimators=200)
    # model = TPOTRegressor(n_jobs=cpu_count(), max_time_seconds=60 * 30, max_eval_time_seconds=60, early_stop=3, verbose=1)
    # fit the model and print metrics
    model.fit(train_x, train_y.to_numpy().ravel())
    model.feature_names_in_ = list(train_x.columns)
    test_pred = pd.Series(model.predict(test_x).ravel(), index=test_x.index)
    train_pred = pd.Series(model.predict(train_x).ravel(), index=train_x.index)

    # also predict on the unfiltered data, so we can feed it to subsequent models
    # TODO does this cause us to use predictions from the training set?
    unfilt_pred = pd.concat(
        [
            pd.Series(
                model.predict(train_x_unfilt[train_x.columns]).ravel(),
                index=train_x_unfilt.index,
            ),
            pd.Series(
                model.predict(test_x_unfilt[train_x.columns]).ravel(),
                index=test_x_unfilt.index,
            ),
        ]
    ).sort_index()
    unfilt_pred.name = model_conf.output if model_conf.output else f"{target_cols[0]}"

    # report accuracy
    r2 = r2_score(test_y, test_pred)
    rmse = root_mean_squared_error(test_y, test_pred)
    logger.info(f"'{target_cols[0]}' {type(model).__name__} model: R2={r2:.3f}, RMSE={rmse:.3f}")
    logger.info(f"Model: {model}")

    # shift input cols back up (shifted down in preprocessing) by the horizon so the plots are aligned
    sample_rate_mins = train_test_df.attrs["sample_rate_mins"]
    horizon_rows = int(model_conf.horizon_mins / sample_rate_mins)
    input_cols = [c for c in train_test_df.columns if c not in target_cols]
    train_test_df[input_cols] = train_test_df[input_cols].shift(-horizon_rows)

    # TODO drop unused modelling code:
    # if isinstance(model, PySRRegressor):
    #     logger.info(f"Symbolic regression models: {model.equations_}")
    # if isinstance(model, TPOTRegressor):
    #     with pd.option_context(
    #         "display.max_rows",
    #         None,
    #         "display.max_columns",
    #         None,
    #         "display.width",
    #         400,
    #         "display.max_colwidth",
    #         50,
    #     ):
    #         logger.info(f"Best TPOT models: \n{model.pareto_front}")
    #         model.pareto_front.to_csv(f"output/{target_cols[0]}_{site}_tpot_pareto_front.csv")
    #         pipeline: Pipeline = model.fitted_pipeline_
    #         with open(f"output/{target_cols[0]}_{site}_tpot_pipeline.pkl", "wb") as f:
    #             pickle.dump(pipeline, f)
    #
    #         # print all hyperparameters
    #         for n in model.fitted_pipeline_.graph.nodes:
    #             print(n, " : ", model.fitted_pipeline_.graph.nodes[n]["instance"])

    if isinstance(model, LinearModel):
        features = dict(zip(np.round(model.coef_, 4), model.feature_names_in_))
        features = dict(sorted(features.items(), reverse=True))
        coeff_str = [f"{k}: {v}" for k, v in features.items()]
        formatted_coeff_str = "\n".join(coeff_str)
        logger.info(f"Model coefficients: \n{formatted_coeff_str}")

    if draw_plots:
        # add predictions to the original df
        title = f"Train and Test Predictions - target={target_cols[0]}, site={site}, model={type(model).__name__} - R2={r2:.3f}, RMSE={rmse:.3f}"
        train_test_df[f"{target_cols[0]}_test_pred"] = test_pred.copy()
        train_test_df[f"{target_cols[0]}_train_pred"] = train_pred.copy()
        fig1 = plot_df(
            train_test_df.drop(columns=lagged_cols).resample(f"{sample_rate_mins}min").first(),
            title,
        )

        # scatterplot of test_pred vs target values
        title = f"Predictions vs Actuals - target={target_cols[0]}, site={site}, model={type(model).__name__}"
        fig2 = px.scatter(x=test_y[target_cols[0]], y=test_pred.ravel(), title=title)
        fig2.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        fig2.data[0].name = "test"
        fig2.add_scatter(
            x=train_y[target_cols[0]],
            y=model.predict(train_x).ravel(),
            mode="markers",
            name="train",
        )

        figs_to_html(
            [fig1, fig2],
            f"output/{title} - R2 score={r2:.3f}, RMSE={rmse:.3f}",
            show=show_plots,
        )

    # copy all attrs from the original df to the model
    model.attrs = train_test_df.attrs

    return model, unfilt_pred


if __name__ == "__main__":
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=200)

    # fixed dates will reuse data caching
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2023, 6, 26)

    site = TrainSite()
    site.run(clayton_config.model_conf, start_date, end_date)
