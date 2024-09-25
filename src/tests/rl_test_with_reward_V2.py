from pathlib import Path
import numpy as np
import pytest
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper, ahu_sa_fan_speed, chiller_elec_power, zone_temp, oa_temp
from dch.utils.init_utils import cd_project_root
from pandas import DataFrame
from pendulum import parse
from pandas import DataFrame, Series
from hvac_gym.gym.gym_for_rl_price_aware_reward import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent
from hvac_gym.sites import newcastle_config
from datetime import datetime
from gymnasium.wrappers import NormalizeObservation
from gym.wrappers import RescaleAction

import math

cd_project_root()
Path("output").mkdir(exist_ok=True)


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


def nonlinear_temperature_response(T):
    T_mid = 10
    s = 5
    scale = 1.5
    return scale * np.tanh((T - T_mid) / s)


def reward_function_flexibility(low_boundary, up_boundary, indoor_temp, outdoor_temp, cooling_power, price, predicted_price):
    weight_edge = 0.001
    min_price = 0.0444 - weight_edge
    max_price = 0.0888 + weight_edge
    normalized_price = 1 * (price - min_price) / (max_price - min_price)
    occupied_period = 0
    load_shaping_mode = 0
    if predicted_price - price > 0:
        load_shaping_mode = 1

    if (low_boundary < 21) and (up_boundary > 24) and (load_shaping_mode == 0):
        ref = low_boundary
    elif (low_boundary < 21) and (up_boundary > 24) and (load_shaping_mode == 1):
        edge = nonlinear_temperature_response(outdoor_temp)
        ref = (up_boundary + low_boundary) / 2 + edge
    else:
        occupied_period = 1
        edge = nonlinear_temperature_response(outdoor_temp)
        ref = (up_boundary + low_boundary) / 2 + edge
    penalty_weight_thermal = 0.5
    penalty_weight_energy = 1 - penalty_weight_thermal
    real_difference = abs(indoor_temp - ref)
    if (occupied_period == 0) and (load_shaping_mode == 0):
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        thermal_reward = reward_(real_difference)
        if low_boundary < indoor_temp < up_boundary:
            scenario = 1
            consumption = cooling_power
            energy_reward = max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 1.1
                consumption = cooling_power
                energy_reward = -max(0, ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 1.2
                consumption = cooling_power
                normalized_outdoor = abs((outdoor_temp - 11) / (37 - 11))
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min))) * (1 + normalized_outdoor)
            final_reward = thermal_reward * penalty_weight_thermal + energy_reward * (penalty_weight_energy)

    elif (occupied_period == 0) and (load_shaping_mode == 1):  # day
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1 - normalized_price
        weight_E = 1 - weight_T
        if low_b_load_shaping <= indoor_temp <= up_b_load_shaping:
            scenario = 2
            thermal_reward = reward_(real_difference)
            consumption = 0
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 2.1
                consumption = cooling_power
                energy_reward = -max(0, ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 2.2
                consumption = cooling_power
                normalized_outdoor = abs((outdoor_temp - 11) / (37 - 11))
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min))) * (1 + normalized_outdoor)
            final_reward = thermal_reward * penalty_weight_thermal + energy_reward * (penalty_weight_energy)
    elif occupied_period == 1:
        weight_T = min((1 - normalized_price), normalized_price)
        weight_E = 1 - weight_T
        if (low_boundary) < indoor_temp < (up_boundary):
            scenario = 3
            thermal_reward = reward_(real_difference)
            consumption = cooling_power  # +fan_action
            energy_reward = max(0, 1 - ((consumption - energy_min) / (1 * energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 3.1
                consumption = cooling_power
                energy_reward = -max(0, ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 3.2
                consumption = cooling_power
                normalized_outdoor = abs((outdoor_temp - 11) / (37 - 11))
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min))) * (1 + normalized_outdoor)
            final_reward = thermal_reward * penalty_weight_thermal + energy_reward * (penalty_weight_energy)

    return final_reward, thermal_reward, energy_reward, ref, weight_T, scenario


class Rl_test:
    now = datetime.now()
    current_time = now.strftime("%dth-%b-%H-%M")

    def test_gym_with_RL(self) -> None:
        time_interval = 10
        start = parse("2022-01-01", tz="Australia/Sydney")
        site_config = newcastle_config.model_conf

        def example_reward_func(observations: Series) -> float:
            indoor_temp = observations.iloc[0]
            outdoor_temp = observations.iloc[1]
            up_boundary = observations.iloc[2]
            low_boundary = observations.iloc[3]
            cooling_power = observations.iloc[4]
            price = observations.iloc[5]
            predicted_price = observations.iloc[6]
            reward, thermal_reward, energy_reward, ref, weight_T, scenario = reward_function_flexibility(
                low_boundary, up_boundary, indoor_temp, outdoor_temp, cooling_power, price, predicted_price
            )
            return reward, thermal_reward, energy_reward, ref, weight_T, scenario

        _number_of_observations = 7
        env = HVACGym(site_config, reward_function=example_reward_func, sim_start_date=start, amount_of_observations=_number_of_observations)

        environment = NormalizeObservation(env)
        print("Action Space:", environment.action_space)
        print("Observation Space:", environment.observation_space)

        training_days = 120
        test_days = 120  # 14
        training_steps = int((60 * 24 / time_interval) * training_days)
        test_steps = int(60 * 24 / time_interval * test_days)
        episodes = 5
        from stable_baselines3 import DQN, A2C, PPO, SAC

        TRAIN_NEW_AGENT = True  # False #True
        MODEL_Name = "CSIRO_gym_agent"
        if TRAIN_NEW_AGENT == True:
            _ent_coef_ = "auto"
            agent = SAC(
                "MlpPolicy", environment, learning_rate=0.0001, batch_size=int(1 * (60 * 24 / time_interval)), ent_coef=_ent_coef_, gamma=0.90
            )  # gamma=0.95
            for episode in range(episodes):
                print("This is the {}th episode".format(episode))
                environment.reset(new_parameter=True)
                agent.learn(total_timesteps=training_steps, reset_num_timesteps=False)

            # agent.learn(total_timesteps=training_steps)
            agent.save(MODEL_Name)
            print("learning progress is completed, now testing is started.")
        else:
            print("Launching the pre-trained agent for demo.")
            agent = SAC.load(MODEL_Name)
            print("Now testing is started.")

        obs = np.zeros(_number_of_observations)
        environment.reset(new_parameter=True)
        for step in range(test_steps):
            try:
                print("now the current step is: ", step)
                # print("--obs--")
                # print(obs)
                # print("--action--")
                action, _ = agent.predict(obs, deterministic=True)  # c
                # print(action)
                # print("**********")
                observation, reward, done, result_sting, _ = environment.step(action)
                obs = observation

            except KeyboardInterrupt:
                raise


if __name__ == "__main__":
    test = Rl_test()
    test.test_gym_with_RL()
