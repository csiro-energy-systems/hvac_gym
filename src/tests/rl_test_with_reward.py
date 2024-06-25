from pathlib import Path
import numpy as np
import pytest
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper, ahu_sa_fan_speed, chiller_elec_power, zone_temp, oa_temp
from dch.utils.init_utils import cd_project_root
from pandas import DataFrame
from pendulum import parse
from pandas import DataFrame, Series
from hvac_gym.gym.gym_for_RL_inherent_reward import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent
from hvac_gym.sites import newcastle_config
from datetime import datetime
from gymnasium.wrappers import NormalizeObservation
from gym.wrappers import RescaleAction
from stable_baselines3 import DQN, A2C, PPO, SAC
import math

cd_project_root()
Path("output").mkdir(exist_ok=True)
initial = 0
result_learning = ""


def optimized_outdoor_diff(x):
    beta = 1
    delta = 17.37
    return 1 / (1 + math.exp(-beta * (x - delta)))


def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_(input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a


energy_min = 0
energy_max = 100


def reward_function_thermal_only(low_boundary, up_boundary, indoor_temp, outdoor_temp):
    target = (up_boundary + low_boundary) / 2
    diff_curr = abs(indoor_temp - target)
    if low_boundary < indoor_temp < up_boundary:
        R_ = reward_(diff_curr)
    else:
        R_ = penality_(diff_curr)
    return R_


class RlAgent:
    now = datetime.now()
    current_time = now.strftime("%dth-%b-%H-%M")
    initial = 0

    def test_gym_with_RL(self) -> None:
        time_interval = 10
        """Tests a long simulation of the gym environment"""
        start = parse("2023-01-01", tz="Australia/Sydney")
        site_config = newcastle_config.model_conf

        def example_reward_func(observations: Series) -> float:
            """Example (but fairly pointless) reward function.
            Replace with your own bespoke reward function calculated from anything in the observations."""
            indoor_temp = observations.iloc[0]
            outdoor_temp = observations.iloc[1]
            up_boundary = observations.iloc[2]
            low_boundary = observations.iloc[3]
            reward = reward_function_thermal_only(low_boundary, up_boundary, indoor_temp, outdoor_temp)
            return reward

        env = HVACGym(site_config, reward_function=example_reward_func, sim_start_date=start)
        environment = NormalizeObservation(env)
        print("Action Space:", environment.action_space)
        print("Observation Space:", environment.observation_space)

        training_days = 150
        test_days = 10
        training_steps = int((60 * 24 / time_interval) * training_days)
        test_steps = int(60 * 24 / time_interval * test_days)
        _ent_coef_ = "auto"
        agent = SAC("MlpPolicy", environment, learning_rate=0.001, batch_size=int(1 * (60 * 24 / time_interval)), ent_coef=_ent_coef_)

        agent.learn(total_timesteps=training_steps)
        print("learning progress is completed, now testing is started.")

        step = 0
        obs = [0, 0, 0, 0]
        environment.reset()
        for step in range(test_steps):
            try:
                # environment.title = agent.name
                print("now the current step is: ", step)
                t0 = datetime.now()
                print("--obs--")
                print(obs)
                print("--action--")
                action, _ = agent.predict(obs, deterministic=True)  # c
                print(action)
                # string_action=str(action[0])+','+str(action[1])+','+str(action[0])+','+str(action[0])+'\n'
                print("**********")
                observation, reward, done, result_sting, _ = environment.step(action)
                obs = observation

            except KeyboardInterrupt:
                raise


if __name__ == "__main__":
    test = RlAgent()
    test.test_gym_with_RL()
