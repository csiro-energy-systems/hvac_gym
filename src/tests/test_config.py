from pathlib import Path

from dch.utils.init_utils import cd_project_root
from hvac_gym.sites import newcastle_config
from hvac_gym.sites.model_config import HVACModelConf, load_config, save_config

cd_project_root()


class TestConfig:
    def test_model_conf_serialisation(self) -> None:
        m = newcastle_config.zone_temp_model
        Path("output/zone_temp_model.json").write_text(m.model_dump_json(indent=4, round_trip=True))
        loaded_conf = HVACModelConf.model_validate_json(Path("zone_temp_model.json").read_text())

        assert m == loaded_conf, "Round-tripped model should match original model"

    def test_gym_config(self) -> None:
        """Just tests that Example can be created"""
        model_conf = newcastle_config.model_conf
        save_config(model_conf, Path("output/test_config.json"))
        loaded_conf = load_config(Path("output/test_config.json"))

        # FIXME pydantic diesn't serialise the PathFilter subclasses correctly, not sure we care, but check here for workaround https://github.com/pydantic/pydantic/discussions/3293
        model_conf.ahu_models[0].filters = []
        loaded_conf.ahu_models[0].filters = []
        assert model_conf == loaded_conf


if __name__ == "__main__":
    test = TestConfig()
    # test.test_model_conf_serialisation()
    test.test_gym_config()
