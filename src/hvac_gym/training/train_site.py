import pickle
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import plotly.io as pio
import sklearn
from dch.dch_interface import DCHBuilding, DCHInterface
from dch.dch_model_utils import flatten, rename_columns
from dch.paths.dch_paths import SemPath, SemPaths
from dch.utils.data_utils import resample_and_join_streams
from dch.utils.init_utils import cd_project_root
from dotenv import load_dotenv
from joblib import Memory
from pandas import DataFrame
from plotly import express as px
from plotly.graph_objs import Figure
from pydantic import BaseModel, ConfigDict
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from tpot2 import TPOTRegressor
from tqdm import tqdm

from hvac_gym.config.log_config import get_logger
from hvac_gym.sites import newcastle_config
from hvac_gym.sites.model_config import HVACModel, HVACModelConf
from hvac_gym.training.train_utils import split_alternate_days
from hvac_gym.vis.vis_tools import figs_to_html

pio.templates.default = "plotly_dark"

logger = get_logger()
cd_project_root()

pd.options.mode.chained_assignment = None
pd.set_option(
    "display.max_rows",
    10,
    "display.max_columns",
    20,
    "display.width",
    200,
    "display.max_colwidth",
    30,
)


class TrainingData(BaseModel):
    """A set of data and metadata for a set of related streams/points"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # tell pydantic to allow dataframes etc

    data: DataFrame
    streams: DataFrame
    sample_rate: float


class TrainingSet(BaseModel):
    """A set of multiple `TrainingData` input and target instances for a single physical model"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # tell pydantic to allow dataframes etc

    target: TrainingData
    inputs: list[TrainingData]


class TrainSite:
    """For a given site, iterates through all model definitions, gets/preprocesses data for all available AHUs and trains predictive ML models on
    each."""

    def __init__(self) -> None:
        load_dotenv()
        self.dch = DCHInterface()

    def check_streams(
        self, model_conf: HVACModel, building: DCHBuilding
    ) -> dict[str, dict[SemPath, DataFrame]]:
        """Check that necessary streams exist
        :param model_conf: the model configuration
        :param building: the building to check streams for
        :return:
        """
        target_point = model_conf.target
        input_points = model_conf.inputs

        target_stream = self.dch.find_streams_path(
            building, target_point, long_format=True
        )
        input_streams = {
            point: self.dch.find_streams_path(building, point, long_format=True)
            for point in input_points
        }

        missing_streams = (
            [point for point, streams in input_streams.items() if streams.empty]
        ) + ([target_point] if target_stream.empty else [])
        if len(missing_streams) > 0:
            logger.error(
                f"Missing target or input streams for {building}. Missing inputs: {missing_streams}"
            )

        return {
            "target_streams": {target_point: target_stream},
            "input_streams": input_streams,
        }

    def preprocess_data(
        self,
        data_set: TrainingSet,
        site_conf: HVACModelConf,
        model_conf: HVACModel,
        streams: dict[str, DataFrame],
    ) -> DataFrame:
        """Preprocesses the training data for each AHU and model
        :param streams:
        """
        target = data_set.target
        inputs = data_set.inputs

        # concat input all streams into a dataframe
        building_input_streams = pd.concat([i.streams for i in inputs])
        building_cols = list(flatten(building_input_streams["column_name"]))
        building_df, sample_rate = resample_and_join_streams(
            [i.data[[c for c in building_cols if c in i.data.columns]] for i in inputs]
        )

        # take median of each group of building data types (e.g. there's often several OAT sensors)
        building_stream_types = building_input_streams.groupby("type_path").aggregate(
            {"column_name": lambda x: list(x)}
        )
        for stream_type in building_stream_types.index:
            cols = [
                c
                for c in building_stream_types.loc[stream_type, "column_name"]
                if c in building_df.columns
            ]
            building_df[f"{stream_type}_median"] = building_df[cols].median(
                axis=1, skipna=True
            )
            # building_df[f"{stream_type}_total"] = building_df[cols].sum(axis=1, skipna=True)
            building_df = building_df.drop(columns=cols)

        input_cols = list(building_df.columns)
        target_col = model_conf.target

        # append the median of the target data to the building data
        target_cols = list(target.streams["column_name"])
        target_df = target.data[target_cols]
        target_df = pd.DataFrame(target_df.median(axis=1), columns=[target_col])
        building_df = target_df.resample(f"{sample_rate}min").median().join(building_df)

        # add lagged columns for each input
        lagged_cols = []
        for col in input_cols:
            for lag in model_conf.lags:
                lag_col = f"{col}+{lag * sample_rate}min"
                building_df[lag_col] = building_df[col].shift(lag)
                lagged_cols.append(lag_col)

        # add metadata to the dataframe
        building_df.attrs["lagged_cols"] = lagged_cols
        building_df.attrs["input_cols"] = input_cols
        building_df.attrs["target_col"] = target_col
        building_df.attrs["sample_rate_mins"] = target.sample_rate
        building_df.attrs["streams"] = streams

        # shift (non-target) inputs down by the horizon so they are used to forecast future targets
        horizon_mins = model_conf.horizon_mins
        horizon_rows = int(horizon_mins / target.sample_rate)
        input_cols = [c for c in building_df.columns if c != target_col]
        building_df[input_cols] = building_df[input_cols].shift(horizon_rows)

        return building_df

    def train_site_model(
        self,
        target_cols: list[str],
        train_test_df: DataFrame,
        site: DCHBuilding,
        model_conf: HVACModelConf,
    ) -> RegressorMixin:

        # TODO make this generic
        # train_test_df = train_test_df.query("Valve_Position_Sensor_total > 0")

        _input_cols = train_test_df.attrs["input_cols"]
        lagged_cols = train_test_df.attrs["lagged_cols"]
        _sample_rate = train_test_df.attrs["sample_rate_mins"]

        train_test_df = train_test_df.sort_index().dropna()
        train, test = split_alternate_days(train_test_df, n_sets=2)

        train_x, train_y = train.drop(columns=target_cols), train[target_cols]
        test_x, test_y = test.drop(columns=target_cols), test[target_cols]

        # linear models
        # model = sklearn.linear_model.LinearRegression()
        model = sklearn.linear_model.ElasticNetCV(n_jobs=-1)  # very fast fitting
        # model = sklearn.linear_model.LassoCV(n_jobs=-1)

        # nonlinear models
        # model = ELMRegressor(**{'n_neurons': 119,'ufunc': 'sigm','alpha': 0.011,'include_original_features': True,'density': 0.7,
        # 'pairwise_metric': 'euclidean'})
        # model, fig2 = elm_optuna_param_search(train_x, train_y, n_trials=100)
        # model = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, verbose=0, n_estimators=30)
        # model = ExtraTreesRegressor(n_jobs=-1, n_estimators=200)
        # model = TPOTRegressor(n_jobs=cpu_count(), max_time_seconds=60 * 30, max_eval_time_seconds=60, early_stop=3, verbose=1)

        # fit the model and print metrics
        model.fit(train_x, train_y)
        test_pred = pd.Series(model.predict(test_x).ravel(), index=test_x.index)
        train_pred = pd.Series(model.predict(train_x).ravel(), index=train_x.index)
        r2 = r2_score(test_y, test_pred)
        rmse = root_mean_squared_error(test_y, test_pred)
        logger.info(
            f"'{target_cols[0]}' {type(model).__name__} model: R2={r2:.3f}, RMSE={rmse:.3f}"
        )

        # shift input cols back up (shifted down in preprocessing) by the horizon so the plots are aligned
        sample_rate_mins = train_test_df.attrs["sample_rate_mins"]
        horizon_rows = int(model_conf.horizon_mins / sample_rate_mins)
        input_cols = [c for c in train_test_df.columns if c not in target_cols]
        train_test_df[input_cols] = train_test_df[input_cols].shift(-horizon_rows)

        if isinstance(model, TPOTRegressor):
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                400,
                "display.max_colwidth",
                50,
            ):
                # logger.info(f"TPOT All Evaluated Individuals: \n{model.evaluated_individuals}")
                logger.info(f"Best TPOT models: \n{model.pareto_front}")
                model.pareto_front.to_csv(
                    f"output/{target_cols[0]}_{site}_tpot_pareto_front.csv"
                )
                pipeline: Pipeline = model.fitted_pipeline_
                pickle.dump(
                    pipeline,
                    open(f"output/{target_cols[0]}_{site}_tpot_pipeline.pkl", "wb"),
                )
                # print all hyperparameters
                for n in model.fitted_pipeline_.graph.nodes:
                    print(n, " : ", model.fitted_pipeline_.graph.nodes[n]["instance"])

        if isinstance(model, LinearModel):
            features = dict(zip(np.round(model.coef_, 4), model.feature_names_in_))
            features = dict(sorted(features.items(), reverse=True))
            logger.info(f"Model coefficients: \n{pformat(features, width=200)}")

        # add predictions to the original df
        train_test_df[f"{target_cols[0]}_test_pred"] = test_pred.copy()
        train_test_df[f"{target_cols[0]}_train_pred"] = train_pred.copy()
        fig1 = self.plot_df(
            train_test_df.drop(columns=lagged_cols),
            f"Train and Test Predictions - {target_cols[0]}_{site}, {type(model).__name__} - R2={r2:.3f}, RMSE={rmse:.3f}",
        )

        # scatterplot of test_pred vs target values
        title = (
            f"Predictions vs Actuals  - {target_cols[0]}_{site}, {type(model).__name__}"
        )
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
            show=False,
        )

        # copy all attrs from the original df to the model
        model.attrs = train_test_df.attrs

        return model

    def plot_df(self, df: DataFrame, title: str, out_dir: Path | None = None) -> Figure:
        fig = px.line(df, height=1600)
        fig.update_layout(title=title)
        if out_dir:
            figs_to_html([fig], out_dir / f"{title}.html", show=True)
        return fig

    def run(self, site_config: HVACModelConf, start: datetime, end: datetime) -> None:
        """Main entrypoint to acquire data, train models and run gym-like simulation for a site"""
        models: dict[SemPath | SemPaths, RegressorMixin] = {}
        model_dfs = []
        for model_conf in site_config.ahu_models:
            logger.info(
                f"Training model {model_conf.target} model for {site_config.site}"
            )

            building = site_config.site

            streams = self.check_streams(model_conf, building)
            streams = self.validate_streams(streams)
            data: TrainingSet = get_data(streams, building, start, end)

            building_df = self.preprocess_data(data, site_config, model_conf, streams)
            model = self.train_site_model(
                [model_conf.target], building_df, site_config.site, model_conf
            )
            models[model_conf.target] = model
            pickle.dump(
                model,
                open(f"output/{site_config.site}_{model_conf.target}_model.pkl", "wb"),
            )

            # save the preprocessed DF from each model.  Inputs are shifted so ready for prediction.
            model_dfs.append(building_df)

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

        # Step the models through a simulation period. Predict with each in the specified order then update the dataframe with predictions
        self.simulate(sim_df, models, streams, site_config)

    def simulate(
        self,
        sim_df: DataFrame,
        models: dict[SemPath | SemPaths, RegressorMixin],
        streams: dict[str, dict[SemPath | SemPaths, DataFrame]],
        site_config: HVACModel,
    ) -> None:
        """
        Simple gym-like simulation with the models and combined dataframe.
        :param sim_df: the combined dataframe with series for all models inputs/targets
        :param models: a dict of target:train-model pairs
        :param site_config: the site configuration
        """

        sim_df = sim_df.dropna(how="any")
        sim_df.columns = [str(c) for c in sim_df.columns]
        # extract target actuals for plotting later
        targets = [str(m.target) for m in site_config.ahu_models]
        actuals = sim_df[targets].copy()
        actuals.columns = [f"{c}_actual" for c in actuals.columns]
        sim_df = actuals.join(sim_df)  # for debugging

        for idx, time in enumerate(
            tqdm(sim_df.index[:200], desc="Simulating", unit="steps")
        ):

            all_inputs_no_lags = []

            for model_conf in site_config.ahu_models:
                target = model_conf.target

                model = models[target]
                inputs = model.feature_names_in_
                model_df = sim_df[inputs]

                inputs_no_lags = model.attrs["input_cols"]
                all_inputs_no_lags.extend(inputs_no_lags)

                # set chilled water valve to square wave, cycling every N steps
                chwv_col = "AHU|Chilled_Water_Coil|Chilled_Water_Valve|Valve_Position_Sensor_median"
                cycle_steps = 12
                chwv_sp = 100 if idx % cycle_steps < cycle_steps / 2 else 0
                # logger.info(f"Setting {chwv_col} to {chwv_sp}")
                sim_df.loc[time, chwv_col] = chwv_sp
                # TODO set lags also

                # get the model's prediction for the next timestep.
                # note: inputs/lags have already been down-shifted, so we can just apply prediction to the inputs at the timestamp directly.
                # this prediction will _apply_ to timestamp + horizon_mins for the model though
                predict_df = model_df.loc[[time]]

                # Run the prediction and update the building_df with result
                prediction = model.predict(predict_df)
                predict_time = time + timedelta(minutes=model_conf.horizon_mins)
                sim_df.loc[predict_time, str(target)] = prediction

        # plot the simulation results with lines
        title = f"Simulation results for {site_config.site}"
        p = sim_df[targets + all_inputs_no_lags + list(actuals.columns)].melt(
            ignore_index=False
        )
        fig = px.line(p, x=p.index, y="value", color="variable", title=title)
        figs_to_html([fig], f"output/{title}", show=True)

    def validate_streams(self, streams: dict[str, DataFrame]) -> dict[str, DataFrame]:
        target_streams = list(streams["target_streams"].values())[0]
        input_streams = streams["input_streams"]

        def append_ahu_name(streams: DataFrame) -> DataFrame:
            streams["ahu_point"] = (
                streams["type_path"].str.split("|").apply(lambda x: x[0] == "AHU")
            )
            streams["ahu_name"] = streams.apply(
                lambda x: x["name_path"].split("|")[0] if x["ahu_point"] else None,
                axis=1,
            )
            return streams

        append_ahu_name(target_streams)
        for point in input_streams.keys():
            append_ahu_name(input_streams[point])

        target_ahus = set(target_streams["ahu_name"].to_list())
        logger.info(f"Found target AHUs: {target_ahus}")

        input_ahu_dict = {
            point: (
                set(input_streams[point]["ahu_name"])
                if "ahu_name" in input_streams[point].columns
                else None
            )
            for point in input_streams.keys()
        }

        input_ahus = [set(ahus) for ahus in input_ahu_dict.values() if ahus is not None]

        # check that there are common AHUs between target and all the input streams
        _common_ahus = set(target_ahus).intersection(*input_ahus)

        return streams


memory = Memory(Path.home() / ".pyfunc_cache")
try:
    memory.reduce_size(bytes_limit="1G", age_limit=timedelta(days=365))
except ValueError:
    pass  # workaround a bug when cache is empty


@memory.cache()
def get_data(
    streams: dict[str, DataFrame],
    building: DCHBuilding,
    start: datetime,
    end: datetime,
) -> TrainingSet:
    """
    Gets a dictionary of dataframes for the target and input streams
    Each
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
    target_streams["sem_path"] = [target_path] * len(
        target_streams
    )  # add column containing SemPath instance for later use
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
    ]

    # rename data columns using brick classes, and add streams to inputs dict
    for i, point in enumerate(streams["input_streams"].keys()):
        inputs[i]["data"], inputs[i]["streams"] = rename_columns(
            inputs[i]["data"], streams["input_streams"][point]
        )

        # add column containing SemPath instances for later use
        inputs[i]["streams"]["sem_path"] = [point] * len(inputs[i]["streams"])

        # remove any stream rows that we don't have data for
        inputs[i]["streams"] = inputs[i]["streams"].query(
            "column_name in @inputs[@i]['data'].columns"
        )

    input_data = [
        TrainingData(data=i["data"], streams=i["streams"], sample_rate=i["sample_rate"])
        for i in inputs
    ]

    training_set = TrainingSet(target=target_data, inputs=input_data)

    return training_set


if __name__ == "__main__":
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=200)

    # fixed dates to reuse data caching
    start_date = datetime(2023, 10, 7)
    end_date = datetime(2024, 4, 24)

    site = TrainSite()
    site.run(newcastle_config.model_conf, start_date, end_date)
    # site.run(clayton_config.model_conf, start_date, end_date)
