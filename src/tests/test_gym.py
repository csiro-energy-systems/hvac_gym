import pytest

from hvac_gym.sites import newcastle_config
import numpy as np

from hvac_gym.gym.gym import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent


class TestGym:
    def test_gym_step(self) -> None:
        """Tests a single step of teh gym environment"""
        from hvac_gym.sites import newcastle_config

        site_config = newcastle_config.model_conf
        env = HVACGym(site_config)
        env.reset()
        env.step(np.array([0, 0, 0]))
        env.render()
        env.close()

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
