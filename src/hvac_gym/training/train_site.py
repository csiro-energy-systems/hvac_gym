from datetime import datetime, timedelta

import pandas as pd
from config.log_config import get_logger
from dch.dch_interface import DCHBuilding, DCHInterface
from dch.utils.init_utils import cd_project_root
from dotenv import load_dotenv
from pandas import DataFrame

from hvac_gym.sites import clayton_config, newcastle_config
from hvac_gym.sites.model_config import HVACModelConf

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


class TrainSite:

    def __init__(self) -> None:
        load_dotenv()
        self.dch = DCHInterface()

    def check_streams(
        self, model_conf: HVACModelConf, building: DCHBuilding
    ) -> dict[str, DataFrame]:
        """Check that necessary streams exist
        :param model_conf: the model configuration
        :param building: the building to check streams for
        :return:
        """
        target_point = model_conf.target
        input_points = model_conf.inputs

        target_stream = self.dch.find_streams_path(building, target_point)
        input_streams = {
            point: self.dch.find_streams_path(building, point) for point in input_points
        }
        n_input_streams = sum([len(streams) for streams in input_streams.values()])

        missing_streams = (
            [point for point, streams in input_streams.items() if streams.empty]
        ) + ([target_point] if target_stream.empty else [])
        if len(missing_streams) > 0:
            logger.error(
                f"Missing target or input streams for {building}. Missing inputs: {missing_streams}"
            )
        logger.success(
            f"Found all necessary stream IDs ({n_input_streams} input + {len(target_stream)} target) for {building}, {model_conf.target} model"
        )
        return {"target_streams": target_stream, "input_streams": input_streams}

    def get_data(
        self,
        streams: dict[str, DataFrame],
        building: DCHBuilding,
        start: datetime,
        end: datetime,
    ) -> dict[str, DataFrame]:
        target = self.dch.get_df_by_stream_id(
            streams["target_streams"]["streams"].explode().unique().tolist(),
            building=building,
            start=start,
            end=end,
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
        return {"target_df": target, "input_df": inputs}

    def run(self, site_config: HVACModelConf, start: datetime, end: datetime) -> None:
        for model_conf in site_config.ahu_models:
            logger.info(
                f"Training model {model_conf.target} model for {site_config.site}"
            )

            building = site_config.site

            streams = self.check_streams(model_conf, building)
            streams = self.validate_streams(streams)
            data = self.get_data(streams, building, start, end)

            # TODO: implement the following methods
            # self.clean_data()
            # self.train_model()
            # self.validate_model()

            print(data)

    def validate_streams(self, streams: dict[str, DataFrame]) -> dict[str, DataFrame]:
        target_meta = streams["target_streams"]
        input_meta = streams["input_streams"]

        target_ahus = set(target_meta["AHU"].to_list())
        input_ahu_dict = {
            point: (
                set(input_meta[point]["AHU"])
                if "AHU" in input_meta[point].columns
                else None
            )
            for point in input_meta.keys()
        }

        input_ahus = [set(ahus) for ahus in input_ahu_dict.values() if ahus is not None]

        # check that there are common AHUs between target and all the input streams
        _common_ahus = set(target_ahus).intersection(*input_ahus)

        return streams


if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7 * 4)
    site = TrainSite()
    site.run(clayton_config.model_conf, start_date, end_date)
    site.run(newcastle_config.model_conf, start_date, end_date)
