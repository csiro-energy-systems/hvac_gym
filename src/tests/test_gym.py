from pathlib import Path

import pytest
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper
from dch.utils.init_utils import cd_project_root
from pandas import DataFrame

from hvac_gym.sites import newcastle_config
import numpy as np

from hvac_gym.gym.gym import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent

cd_project_root()
Path("output").mkdir(exist_ok=True)


class TestGym:
    @pytest.mark.integration
    def test_gym_step(self) -> None:
        """Tests a single step of teh gym environment"""
        from hvac_gym.sites import newcastle_config

        site_config = newcastle_config.model_conf
        env = HVACGym(site_config)
        env.reset()
        action = DataFrame(
            {
                str(ahu_chw_valve_sp): 0,
                str(ahu_hw_valve_sp): 0,
                str(ahu_oa_damper): 0,
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

        env = HVACGym(site_config)
        env.reset()

        agent = MinMaxCoolAgent(env, cycle_steps=100, cool_chwv_setpoint=100)
        run_gym_with_agent(env, agent, site_config, max_steps=10, show_plot=False)

    @pytest.mark.manual("Long run, manual launch only")
    def test_gym_simulate_long(self) -> None:
        """Tests a long simulation of the gym environment"""
        max_steps = 1000
        site_config = newcastle_config.model_conf

        env = HVACGym(site_config)
        env.reset()

        agent = MinMaxCoolAgent(env, cycle_steps=100, cool_chwv_setpoint=100)
        run_gym_with_agent(env, agent, site_config, max_steps=max_steps, show_plot=False)


if __name__ == "__main__":
    test = TestGym()
    test.test_gym_simulate_long()
