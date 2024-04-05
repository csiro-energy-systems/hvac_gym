from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from config.log_config import get_logger
from dch.dch_interface import DCHBuilding, DCHInterface
from dch.dch_model_utils import flatten, rename_columns
from dch.utils.data_utils import resample_and_join_streams
from dch.utils.init_utils import cd_project_root
from dotenv import load_dotenv
from pandas import DataFrame
from plotly import express as px
from points.dch_point import SemPath
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from hvac_gym.sites import clayton_config, newcastle_config
from hvac_gym.sites.model_config import HVACModel, HVACModelConf
from hvac_gym.vis.vis_tools import df_to_html, figs_to_html

logger = get_logger()

cd_project_root()
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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: DataFrame
    streams: DataFrame
    sample_rate: float


class TrainingSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target: TrainingData
    inputs: list[TrainingData]


class TrainSite:

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

    def get_data(
        self,
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

        target = self.dch.get_df_by_stream_id(
            target_streams["streams"].explode().unique().tolist(),
            building=building,
            start=start,
            end=end,
        )
        target["data"], target_streams = rename_columns(target["data"], target_streams)
        target_streams["sem_path"] = [target_path] * len(
            target_streams
        )  # add column containing SemPath instance for later use

        target_data = TrainingData(
            data=target["data"],
            streams=target_streams,
            sample_rate=target["sample_rate"],
        )

        inputs = [
            self.dch.get_df_by_stream_id(
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

        input_data = [
            TrainingData(
                data=i["data"], streams=i["streams"], sample_rate=i["sample_rate"]
            )
            for i in inputs
        ]

        training_set = TrainingSet(target=target_data, inputs=input_data)

        return training_set

    def preprocess_data(
        self, data_set: TrainingSet, site_conf: HVACModelConf, model_conf: HVACModel
    ) -> None:
        """Preprocesses the training data for each AHU and model"""
        target = data_set.target
        inputs = data_set.inputs

        # streams not associated with an AHU (because they apply to the whole building)
        building_input_streams = pd.concat(
            [i.streams[pd.isna(i.streams["ahu_name"])] for i in inputs]
        )
        building_cols = list(flatten(building_input_streams["column_name"]))
        building_df = pd.concat(
            [i.data[[c for c in building_cols if c in i.data.columns]] for i in inputs]
        )

        # take median of each group of building data types (e.g. there's often several OAT sensors)
        building_stream_types = building_input_streams.groupby("type").aggregate(
            {"column_name": lambda x: list(x)}
        )
        for stream_type in building_stream_types.index:
            cols = [
                c
                for c in building_stream_types.loc[stream_type, "column_name"]
                if c in building_df.columns
            ]
            building_df[stream_type] = building_df[cols].median(axis=1)
            building_df = building_df.drop(columns=cols)

        # streams which apply to any specific AHU
        input_streams = pd.concat(
            [i.streams[~pd.isna(i.streams["ahu_name"])] for i in inputs]
        )

        ahu_names = target.streams["ahu_name"].unique()
        for ahu in tqdm(ahu_names, desc="Preprocessing AHUs", unit="AHUs"):

            # find just the streams and data that apply to this AHU (or all AHUs, like OA Temp)
            ahu_target_streams = target.streams[target.streams["ahu_name"] == ahu]
            ahu_input_streams = input_streams[input_streams["ahu_name"] == ahu]

            target_cols = list(ahu_target_streams["column_name"])
            input_cols = list(flatten(ahu_input_streams["column_name"]))

            required_input_paths = set(
                [*ahu_input_streams["sem_path"], *building_input_streams["sem_path"]]
            )
            missing_sempaths = [
                p for p in model_conf.inputs if p not in required_input_paths
            ]
            if len(missing_sempaths) > 0:
                logger.error(f"Missing input points for AHU {ahu}: {missing_sempaths}")
                continue

            if not any([c in target.data.columns for c in target_cols]):
                logger.error(
                    f"Target columns {target_cols} not found in target data for AHU {ahu}"
                )
                continue

            if not any([c in i.data.columns for c in input_cols for i in inputs]):
                logger.error(
                    f"Input columns {input_cols} not found in input data for AHU {ahu}"
                )
                continue

            logger.success(
                f"Found streams for all {len(required_input_paths)} necessary paths IDs for building, AHU {ahu}, model: {model_conf.target} model."
            )

            # Get the target dataframe
            target_df = target.data[target_cols]

            # Get and join all the dataframes for all columns that match the input_cols
            input_dfs = [
                i.data[[c for c in input_cols if c in i.data.columns]] for i in inputs
            ]
            train_df, sample_rate_mins = resample_and_join_streams(
                [target_df] + input_dfs + [building_df]
            )

            # plot and save data
            out_dir = Path("output")
            px.defaults.template = "plotly"
            fig = px.line(train_df, height=800)
            fig.update_layout(
                title=f"Training Data for {site_conf.site} model for AHU {ahu}"
            )
            all_streams = pd.concat([ahu_target_streams, ahu_input_streams])
            all_streams["sem_path"] = all_streams["sem_path"].astype(str)
            streams_html = df_to_html(all_streams)
            f = f"train_data.{site_conf.site}_{ahu}.html"
            figs_to_html([fig], out_dir / f, extra_html=streams_html, show=False)

    def run(self, site_config: HVACModelConf, start: datetime, end: datetime) -> None:
        for model_conf in site_config.ahu_models:
            logger.info(
                f"Training model {model_conf.target} model for {site_config.site}"
            )

            building = site_config.site

            streams = self.check_streams(model_conf, building)
            streams = self.validate_streams(streams)
            data: TrainingSet = self.get_data(streams, building, start, end)

            # TODO: implement the following methods
            # self.clean_data()
            self.preprocess_data(data, site_config, model_conf)
            # self.validate_model()

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


if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    site = TrainSite()
    site.run(clayton_config.model_conf, start_date, end_date)
    site.run(newcastle_config.model_conf, start_date, end_date)
