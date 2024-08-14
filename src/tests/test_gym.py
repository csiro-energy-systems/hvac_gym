# The Software is copyright (c) CSIRO ABN 41 687 119 230
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

cd_project_root()
Path("output").mkdir(exist_ok=True)

import plotly.io as pio

pio.templates.default = "plotly_dark"


class TestGym:
    @pytest.mark.integration
    def test_gym_step(self) -> None:
        """Tests a single step of teh gym environment"""
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
        max_steps = 500
        chwv_sp = 100
        valve_on_off_hours = 12
        start = parse("2023-10-01", tz="Australia/Sydney")

        site_config = newcastle_config.model_conf

        def example_reward_func(observations: Series) -> float:
            """Example (but fairly pointless) reward function.
            Replace with your own bespoke reward function calculated from anything in the observations."""
            return float(observations[str(chiller_elec_power)] + observations[str(zone_temp)])

        env = HVACGym(site_config, reward_function=example_reward_func, sim_start_date=start)
        agent = MinMaxCoolAgent(env, cycle_steps=6 * valve_on_off_hours, cool_chwv_setpoint=chwv_sp)
        obs, rewards = run_gym_with_agent(env, agent, site_config, max_steps=max_steps, show_plot=False)
        obs_df = pd.concat(obs, axis=1).T
        obs_df["reward"] = rewards

        logger.info(f"Observations: {obs_df}")
        logger.info(f"Total reward was {sum(rewards):.1f}")


if __name__ == "__main__":
    test = TestGym()
    test.test_gym_simulate_long()
