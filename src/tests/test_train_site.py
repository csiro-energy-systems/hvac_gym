from datetime import datetime

import pytest

from hvac_gym.sites import newcastle_config
from hvac_gym.sites.newcastle_config import ahu_chws_elec_power_model
from hvac_gym.training.train_site import TrainSite


class TestTrainSite:
    @pytest.mark.integration
    def test_train_site(self) -> None:
        """Minimal training run.  Requires DCH access to run."""
        start_date = datetime(2024, 7, 1)
        end_date = datetime(2024, 7, 2)
        conf = newcastle_config.model_conf
        conf.out_dir = "temp"
        conf.ahu_models = [ahu_chws_elec_power_model]

        site = TrainSite()
        site.run(conf, start_date, end_date)


if __name__ == "__main__":
    TestTrainSite().test_train_site()
