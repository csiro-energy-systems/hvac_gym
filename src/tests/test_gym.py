# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.

from pathlib import Path
from pprint import pprint

import pandas as pd
import pytest
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper, ahu_sa_fan_speed, chiller_elec_power, zone_temp
from dch.utils.init_utils import cd_project_root
from loguru import logger
from pandas import DataFrame, Series
from pendulum import parse

from hvac_gym.gym.gym import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent
from hvac_gym.sites import newcastle_config
import zipfile
import tempfile

cd_project_root()
Path("output").mkdir(exist_ok=True)

import plotly.io as pio

pio.templates.default = "plotly_dark"


class TestGym:
    def test_with_example_data(self) -> None:
        """Tests the gym environment can simulate with the sample data/models"""

        data = Path("data/sample-models.zip")
        if not data.exists():
            logger.error(f"Data file {data} not found, skipping test")
            return

        site_config = newcastle_config.model_conf.copy()

        # unzip the example data to a temp dir, and run a gym/agent using it
        with zipfile.ZipFile(data, "r") as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                print(f"Extracted to {tmpdirname}")
                pprint(list(Path(tmpdirname).rglob("*")))
                site_config.out_dir = str(Path(tmpdirname) / "output")

                sim_steps = 20
                env = HVACGym(site_config, reward_function=lambda x: 0.0)
                agent = MinMaxCoolAgent(env, cycle_steps=100, cool_chwv_setpoint=100)
                observations, rewards = run_gym_with_agent(env, agent, site_config, max_steps=sim_steps, show_plot=False)
                observations_df = pd.concat(observations, axis=1).T
                assert len(observations_df) == sim_steps
                assert len(rewards) == sim_steps

    @pytest.mark.integration
    def test_gym_step(self) -> None:
        """Tests a single step of the gym environment"""
        from hvac_gym.sites import newcastle_config

        site_config = newcastle_config.model_conf
        env = HVACGym(site_config, reward_function=lambda x: 0.0)
        env.reset()
        action = DataFrame(
            {
                str(ahu_chw_valve_sp): 0,
                str(ahu_hw_valve_sp): 0,
                str(ahu_oa_damper): 0,
                str(ahu_sa_fan_speed): 0,
            },
            index=[0],
        )
        env.step(action)
        env.render()
        env.close()

    @pytest.mark.integration
    def test_gym_simulate(self) -> None:
        """Tests a short simulation of the gym environment"""

        site_config = newcastle_config.model_conf

        env = HVACGym(site_config, reward_function=lambda x: 0.0)
        agent = MinMaxCoolAgent(env, cycle_steps=100, cool_chwv_setpoint=100)
        run_gym_with_agent(env, agent, site_config, max_steps=10, show_plot=False)

    @pytest.mark.manual("Long run, manual launch only")
    @pytest.mark.integration
    def test_gym_simulate_long(self) -> None:
        """Tests a long simulation of the gym environment"""
        max_steps = 24 * 6
        chwv_sp = 100
        valve_on_off_hours = 12
        # start = parse("2023-11-01", tz="Australia/Sydney")
        start = parse("2023-12-14", tz="Australia/Sydney")  # 37°C day

        site_config = newcastle_config.model_conf

        def example_reward_func(observations: Series) -> float:
            """Example (but fairly pointless) reward function.
            Replace with your own bespoke reward function calculated from anything in the observations."""
            return float(observations[str(chiller_elec_power)] + observations[str(zone_temp)])

        env = HVACGym(site_config, reward_function=example_reward_func, sim_start_date=start)
        agent = MinMaxCoolAgent(env, cycle_steps=6 * valve_on_off_hours, cool_chwv_setpoint=chwv_sp)
        obs, rewards = run_gym_with_agent(env, agent, site_config, max_steps=max_steps, show_plot=True)
        obs_df = pd.concat(obs, axis=1).T
        obs_df["reward"] = rewards

        logger.info(f"Observations: \n{obs_df}")
        logger.info(f"Total reward was {sum(rewards):.1f}")  # type: ignore


if __name__ == "__main__":
    test = TestGym()
    test.test_gym_simulate_long()
