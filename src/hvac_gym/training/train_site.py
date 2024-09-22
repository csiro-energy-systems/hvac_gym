# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio
from dch.dch_interface import DCHBuilding, DCHInterface
from dch.dch_model_utils import flatten, rename_columns
from dch.paths.dch_paths import SemPath
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp
from dch.utils.data_utils import resample_and_join_streams
from dch.utils.init_utils import cd_project_root
from dotenv import load_dotenv
from joblib import Memory
from pandas import DataFrame, Series
from plotly import express as px
from plotly.graph_objs import Figure
from pydantic import BaseModel, ConfigDict
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm import tqdm

from hvac_gym.config.log_config import get_logger
from hvac_gym.sites import newcastle_config
from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf
from hvac_gym.training.train_utils import split_alternate_days
from hvac_gym.vis.vis_tools import figs_to_html

pio.templates.default = "plotly_dark"

logger = get_logger()
cd_project_root()

draw_plots = True  # enables/disables plotting during training
show_plots = False  # enables/disables opening plots when draw_plots==True

pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 10, "display.max_columns", 20, "display.width", 200, "display.max_colwidth", 30)

func_cache_dir = Path("output/.func_cache")
memory = Memory(func_cache_dir, verbose=1)
memory.reduce_size(bytes_limit="1G", age_limit=timedelta(days=365))
logger.info(f"Using function cache dir at: {func_cache_dir}")


class TrainingData(BaseModel):
    """A set of data and metadata for a set of related streams/points"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # tell pydantic to allow dataframes etc

    data: DataFrame
    streams: DataFrame
    sample_rate: float


class TrainingSet(BaseModel):
    """A set of multiple `TrainingData` input and target instances for a single physical model"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # tell pydantic to allow dataframes etc

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
        """Check that necessary streams exist
        :param model_conf: the model configuration
        :param building: the building to check streams for
        :return:
        """
        target_point = model_conf.target
        input_points = model_conf.inputs

        target_stream = self.dch.find_streams_path(building, target_point, long_format=True)
        input_streams = {point: self.dch.find_streams_path(building, point, long_format=True) for point in input_points}

        missing_streams = ([point for point, streams in input_streams.items() if streams.empty]) + ([target_point] if target_stream.empty else [])
        if len(missing_streams) > 0:
            logger.error(f"Missing target or input streams for {building}. Missing inputs: {missing_streams}")

        return {
            "target_streams": {target_point: target_stream},
            "input_streams": input_streams,
        }

    def preprocess_data(
        self,
        data_set: TrainingSet,
        site_conf: HVACSiteConf,
        model_conf: HVACModelConf,
        streams: dict[str, DataFrame],
    ) -> tuple[DataFrame, dict[str, Any]]:
        """Preprocesses the training data for each AHU and model"""
        target = data_set.target
        inputs = data_set.inputs

        # concat input all streams into a dataframe
        building_input_streams = pd.concat([i.streams for i in inputs])
        building_cols = list(flatten(building_input_streams["column_name"]))
        building_df, sample_rate = resample_and_join_streams([i.data[[c for c in building_cols if c in i.data.columns]] for i in inputs])

        # take median of each group of building data types (e.g. there's often several OAT sensors)
        # use SemPath instances for new aggregated column names
        building_stream_types = building_input_streams.groupby("type_path").aggregate({"column_name": lambda x: list(x)})
        for stream_type in building_stream_types.index:
            cols = [c for c in building_stream_types.loc[stream_type, "column_name"] if c in building_df.columns]
            sempath = building_input_streams.query("type_path == @stream_type")["sem_path"].unique()[0]
            building_df[sempath] = building_df[cols].median(axis=1, skipna=True)
            # building_df[f"{stream_type}_total"] = building_df[cols].sum(axis=1, skipna=True)
            building_df = building_df.drop(columns=cols)

        input_cols = list(building_df.columns)
        target_col = model_conf.target

        # append the median of the target data to the building data
        target_cols = list(target.streams["column_name"])
        target_df = target.data[target_cols]
        target_df = pd.DataFrame(target_df.median(axis=1), columns=[target_col])
        building_df = (
            target_df.resample(f"{sample_rate}min").median().join(building_df).interpolate(limit=3, limit_direction="forward", method="linear")
        )

        # add lagged columns for each input
        lagged_cols = []
        for col in input_cols:
            for lag in model_conf.lags:
                lag_col = f"{col}_{int(lag * sample_rate)}min"
                building_df[lag_col] = building_df[col].shift(lag)
                lagged_cols.append(lag_col)

        # add metadata to the dataframe
        df_attrs: dict[str, Any] = dict()
        df_attrs["lagged_cols"] = lagged_cols
        df_attrs["input_cols"] = input_cols
        df_attrs["target_col"] = target_col
        df_attrs["sample_rate_mins"] = target.sample_rate
        df_attrs["streams"] = streams

        # shift (non-target) inputs down by the horizon so they are used to forecast future targets
        horizon_mins = model_conf.horizon_mins
        horizon_rows = int(horizon_mins / target.sample_rate)
        input_cols = [c for c in building_df.columns if c != target_col]
        building_df[input_cols] = building_df[input_cols].shift(horizon_rows)

        return building_df, df_attrs

    def run(self, site_config: HVACSiteConf, start: datetime, end: datetime) -> None:
        """Main entrypoint to acquire data, train models and run gym-like simulation for a site"""
        models: dict[HVACModelConf, RegressorMixin] = {}
        model_dfs = []
        model_df_attrs = []
        predictions: list[Series] = []

        common_sample_rate = 0

        # First pass: get all the data and metadata for all models, and store back in the configs
        for model_conf in site_config.ahu_models:
            logger.info(f"Training model {model_conf.output} model for {site_config.site}")
            building = site_config.site

            # Gather and preprocess data
            streams = self.check_streams(model_conf, building)
            streams = self.validate_streams(streams)
            data = get_data(streams, building, start, end)
            building_df, df_attrs = self.preprocess_data(data, site_config, model_conf, streams)

            model_conf.dataframe = building_df
            model_conf.df_attrs = df_attrs

            # Work out the maximum sample rate from all the models' DFs, and use that for all models.
            common_sample_rate = max(common_sample_rate, df_attrs["sample_rate_mins"])

        # Second pass: now we have all the data and attributes, train models on them
        for model_conf in site_config.ahu_models:
            building_df = model_conf.dataframe

            # make sure all models use the same sample rate
            model_conf.df_attrs["sample_rate_mins"] = common_sample_rate
            df_attrs = model_conf.df_attrs

            # Train and test models
            model, preds = train_site_model(building_df, df_attrs, site_config.site, model_conf, predictions)
            predictions.append(preds)
            models[model_conf] = model

            # Save the preprocessed DF from each model.  Inputs are shifted so ready for prediction.
            model_dfs.append(building_df)
            model_df_attrs.append(df_attrs)

        sim_df = self.save_models_and_data(models, model_dfs, model_df_attrs, site_config)

        # Step the models through a simulation period. Predict with each in the specified order then update the dataframe with predictions
        self.simulate(sim_df, models, site_config, sim_steps=200)

        # or profile with cProfile
        # import cProfile
        # cProfile.runctx('self.simulate(sim_df, models, site_config)', globals(), locals(), filename='simulate.prof')

    def simulate(
        self,
        sim_df: DataFrame,
        models: dict[HVACModelConf, RegressorMixin],
        site_config: HVACSiteConf,
        sim_steps: int = 500,
    ) -> None:
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

                inputs_no_lags = [str(i) for i in model_conf.derived_inputs] + [str(i) for i in model_conf.inputs]
                all_inputs_no_lags.extend(inputs_no_lags)

                # set chilled water valve to square wave, cycling every N steps
                chwv_col = str(ahu_chw_valve_sp.name)
                _hwv_col = str(ahu_hw_valve_sp.name)
                cycle_steps = 12 * 2

                chwv_sp = 100 if idx % cycle_steps < cycle_steps / 2 else 0
                sim_df.loc[time, chwv_col] = chwv_sp

                # hwv_sp = 100 if idx % cycle_steps > cycle_steps / 2 else 0
                # sim_df.loc[time, hwv_col] = hwv_sp

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

    def validate_streams(self, streams: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Validate the streams and check for common AHUs between target and input streams"""
        target_streams = list(streams["target_streams"].values())[0]
        input_streams = streams["input_streams"]

        def append_ahu_name(streams: DataFrame) -> DataFrame:
            if "type_path" in streams.columns:
                streams["ahu_point"] = streams["type_path"].str.split("|").apply(lambda x: x[0] == "AHU")
            if "name_path" in streams.columns:
                streams["ahu_name"] = streams.apply(
                    lambda x: x["name_path"].split("|")[0] if x["ahu_point"] else None,
                    axis=1,
                )
            return streams

        # Get list of AHUs that the target streams apply to, if any
        target_ahus: set[str] = set()
        append_ahu_name(target_streams)
        if "ahu_name" in target_streams.columns:
            target_ahus = set(target_streams["ahu_name"].to_list())
            logger.trace(f"Found target AHUs: {target_ahus}")

        # Get list of AHUs that the input streams apply to, if any
        for point in input_streams.keys():
            append_ahu_name(input_streams[point])

        input_ahu_dict = {
            point: (set(input_streams[point]["ahu_name"]) if "ahu_name" in input_streams[point].columns else None) for point in input_streams.keys()
        }

        input_ahus = [set(ahus) for ahus in input_ahu_dict.values() if ahus is not None]

        # check if there are common AHUs between target and all the input streams
        # TODO implement per-AHU modelling where possible. This isn't used yet.
        _common_ahus = set(target_ahus).intersection(*input_ahus)

        return streams

    def save_models_and_data(
        self, models: dict[HVACModelConf, RegressorMixin], model_dfs: list[DataFrame], df_attrs: list[dict[str, Any]], site_config: HVACSiteConf
    ) -> DataFrame:
        """
        Combines dataframes from all models into a single DF that can be used for simulation, and saves it and the models to disk.
        :param models: the trained models
        :param model_dfs: the dataframes for each model
        :param site_config: the site configuration
        :return: the combined dataframe
        """
        # save models to disk
        for model_conf, model in models.items():
            model_file = Path(f"{site_config.out_dir}/{site_config.site}_{model_conf.output}_model.pkl")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

        # Combine DFs from all models
        sim_df = pd.concat(model_dfs, axis="columns")

        # Make targets the first columns
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
        sim_df = actuals.join(sim_df)  # add actuals for debugging
        sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]
        sim_df = sim_df.dropna(how="any")

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
    streams: dict[str, DataFrame],
    building: DCHBuilding,
    start: datetime,
    end: datetime,
) -> TrainingSet:
    """
    Gets a dictionary of dataframes for the target and input streams

    :param streams: dictionary of target and input streams to get data for
    :param building: the building to get data for
    :param start: data start date
    :param end: data end date
    :return: A TrainingSet object with data and metadata for all streams
    """
    target_path = list(streams["target_streams"].keys())[0]
    target_streams = streams["target_streams"][target_path]

    dch = DCHInterface()
    target = dch.get_df_by_stream_id(
        target_streams["streams"].explode().unique().tolist(),
        building=building,
        start=start,
        end=end,
    )
    target["data"], target_streams = rename_columns(target["data"], target_streams)
    target_streams["sem_path"] = [str(target_path)] * len(target_streams)  # add column containing SemPath instance for later use
    # drop target_strems rows for which we have no columns
    target_streams = target_streams.query("column_name in @target['data'].columns")

    target_data = TrainingData(
        data=target["data"],
        streams=target_streams,
        sample_rate=target["sample_rate"],
    )

    inputs = [
        dch.get_df_by_stream_id(
            streams["input_streams"][point]["streams"].explode().unique().tolist(),
            building=building,
            start=start,
            end=end,
        )
        for point in streams["input_streams"].keys()
        if "streams" in streams["input_streams"][point]
    ]

    # rename data columns using brick classes, and add streams to inputs dict
    for i, point in tqdm(enumerate(streams["input_streams"].keys()), desc="Getting input data", total=len(inputs)):
        inputs[i]["data"], inputs[i]["streams"] = rename_columns(inputs[i]["data"], streams["input_streams"][point])

        # add column containing SemPath instances for later use
        inputs[i]["streams"]["sem_path"] = [str(point)] * len(inputs[i]["streams"])

        # remove any stream rows that we don't have data for
        inputs[i]["streams"] = inputs[i]["streams"].query("column_name in @inputs[@i]['data'].columns")

    input_data = [TrainingData(data=i["data"], streams=i["streams"], sample_rate=i["sample_rate"]) for i in inputs]

    training_set = TrainingSet(target=target_data, inputs=input_data)

    return training_set


# @memory.cache()
def train_site_model(
    train_test_df: DataFrame,
    df_attrs: dict[str, Any],
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
    :return: the trained model
    """
    target_cols = [model_conf.target.name]
    output_col = model_conf.output if model_conf.output else target_cols[0]
    lagged_cols = df_attrs["lagged_cols"]
    sample_rate_mins = df_attrs["sample_rate_mins"]

    train_test_df = train_test_df.resample(f"{sample_rate_mins}min").median().copy()

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
    # model, fig2 = elm_optuna_param_search(train_x, train_y, n_trials=100)
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
    horizon_rows = int(model_conf.horizon_mins / sample_rate_mins)
    input_cols = [c for c in train_test_df.columns if c not in target_cols]
    train_test_df[input_cols] = train_test_df[input_cols].shift(-horizon_rows)

    if isinstance(model, LinearModel):
        features = dict(zip(np.round(model.coef_, 4), model.feature_names_in_))
        features = dict(sorted(features.items(), reverse=True))
        coeff_str = [f"{k}: {v}" for k, v in features.items()]
        logger.info("Model coefficients: \n" + "\n".join(coeff_str))

    if draw_plots:
        # add predictions to the original df
        title = f"Train and Test Predictions - target={output_col}, site={site}, model={type(model).__name__} - R2={r2:.3f}, RMSE={rmse:.3f}"
        train_test_df[f"{target_cols[0]}_test_pred"] = test_pred.copy()
        train_test_df[f"{target_cols[0]}_train_pred"] = train_pred.copy()
        fig1 = plot_df(
            train_test_df.drop(columns=lagged_cols).resample(f"{sample_rate_mins}min").first(),
            title,
        )

        # scatterplot of test_pred vs target values
        title = f"Predictions vs Actuals - target={output_col}, site={site}, model={type(model).__name__}"
        fig2 = px.scatter(x=test_y[target_cols[0]], y=test_pred, title=title)
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

    return model, unfilt_pred


if __name__ == "__main__":
    # fixed dates will reuse data caching
    start_date = datetime(2023, 10, 7)
    end_date = datetime(2024, 8, 14)

    site = TrainSite()
    site.run(newcastle_config.model_conf, start_date, end_date)
