from pathlib import Path
import numpy as np
import pytest
import pytz
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper, ahu_sa_fan_speed, chiller_elec_power, zone_temp, oa_temp
from dch.utils.init_utils import cd_project_root
from pandas import DataFrame
from pendulum import parse
from pandas import DataFrame, Series
from hvac_gym.gym.gym_for_RL_implementation import HVACGym, run_gym_with_agent  #Price ToU integarated
from hvac_gym.gym.performance_evaluation_metrics import calculate_thermal_discomfort_kpi,calculate_energy_kpi,_ToU_,calculate_cost_kpi
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


def nonlinear_temperature_response(T):
    T_mid = 10
    s = 5
    scale = 1.5
    return scale * np.tanh((T - T_mid) / s)


def reward_function_flexibility(observations):
    # obsertvations = ["current_hour","indoor_temp", "outdoor_temp", "up_boundary", "low_boundary", "cooling_usage", "price"]
    indoor_temp = observations['indoor_temp']
    outdoor_temp = observations['outdoor_temp']
    up_boundary = observations['up_boundary']
    low_boundary = observations['low_boundary']
    cooling_usage = observations['cooling_usage']
    price = observations['price']
    current_hour=observations['current_hour']
    weight_edge = 0.0001
    min_price = _ToU_()['off-peak'] - weight_edge
    max_price = _ToU_()['peak'] + weight_edge
    normalized_price = 1 * (price - min_price) / (max_price - min_price)
    occupied_period = 0
    load_shaping_mode = 0
    if (7 <= current_hour < 8):
        load_shaping_mode = 1

    if (low_boundary < 20) and (up_boundary > 26) and (load_shaping_mode == 0):
        ref = low_boundary
    elif (low_boundary < 20) and (up_boundary > 26) and (load_shaping_mode == 1):
        edge = nonlinear_temperature_response(outdoor_temp)
        ref = (up_boundary + low_boundary) / 2 + edge
    else:
        occupied_period = 1
        edge = nonlinear_temperature_response(outdoor_temp)
        ref = (up_boundary + low_boundary) / 2 + edge
    real_difference = abs(indoor_temp - ref)
    if (occupied_period == 0) and (load_shaping_mode == 0):
        weight_T = 1 * (_ToU_()['off-peak'] - min_price) / (max_price - min_price)
        weight_E = 1 - weight_T
        consumption = cooling_usage #+ heating_usage
        if (low_boundary < indoor_temp < up_boundary) and (consumption < 10):
            thermal_reward = reward_(real_difference)
            scenario = 1
            energy_reward = max(0, 1 - ((consumption - energy_min) / (2 * energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
        elif (low_boundary < indoor_temp < up_boundary) and (consumption >= 10):
            thermal_reward = 0
            scenario = 1.1
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
        else:
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 1.3
                consumption = 0 # we dont have heating in Newcastle gym yet
                energy_reward = -max(0,
                                     1 - ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 1.4
                consumption = cooling_usage
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

    elif (occupied_period == 0) and (load_shaping_mode == 1):  # day
        up_b_load_shaping = 26  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 20  # min((ref + edge), (ref - edge))
        weight_T = 1
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
            if low_b_load_shaping >= indoor_temp:
                scenario = 2.1
                consumption = 0 # we dont have heating in Newcastle gym yet
                energy_reward = -max(0,
                                     1 - ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 2.2
                consumption = cooling_usage
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
    elif occupied_period == 1:
        weight_T = min((1 - normalized_price), normalized_price)
        weight_E = 1 - weight_T
        if low_boundary < indoor_temp < up_boundary:
            thermal_reward = reward_(real_difference)
            scenario = 3
            consumption = cooling_usage # + heating_usage     we dont have heating in Newcastle gym yet
            energy_reward = max(0, 1 - ((consumption - energy_min) / (2 * energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 3.1
                consumption = 0 # we dont have heating in Newcastle gym yet
                energy_reward = -max(0,
                                     1 - ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 3.2
                consumption = cooling_usage
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E


    return final_reward, thermal_reward, energy_reward, ref, weight_T, scenario

def training_setup():
    training_starts = "2022-01-01"
    training_days = 2
    episodes = 2
    return training_starts, training_days, episodes


def testing_setup():
    testing_starts = "2023-01-01"
    testing_days = 1
    return testing_starts, testing_days

class Rl_test:
    now = datetime.now()
    current_time = now.strftime("%dth-%b-%H-%M")
    print(current_time)
    def test_gym_with_RL(self) -> None:
        time_interval = 10
        training_starts, training_days, episodes = training_setup()
        testing_starts, testing_days = testing_setup()
        start = parse(training_starts).astimezone(pytz.timezone("Australia/Sydney"))
        training_steps = int((60 * 24 / time_interval) * training_days)
        test_steps = int(60 * 24 / time_interval * testing_days)
        site_config = newcastle_config.model_conf

        def example_reward_func(observations: Series) -> float:

            reward, thermal_reward, energy_reward, ref, weight_T, scenario = reward_function_flexibility(
                observations
            )
            return reward, thermal_reward, energy_reward, ref, weight_T, scenario

        _number_of_observations = 7
        env = HVACGym(site_config, reward_function=example_reward_func, sim_start_date=start, amount_of_observations=_number_of_observations)

        environment = NormalizeObservation(env)
        print("Action Space:", environment.action_space)
        print("Observation Space:", environment.observation_space)

        from stable_baselines3 import DQN, A2C, PPO, SAC

        TRAIN_NEW_AGENT = True  # False #True
        MODEL_Name = "CSIRO_Newcastle_agent"
        if TRAIN_NEW_AGENT == True:
            _ent_coef_ = "auto"
            agent = SAC(
                "MlpPolicy", environment, learning_rate=0.0003, batch_size=int(1 * (60 * 24 / time_interval)), ent_coef=_ent_coef_, gamma=0.90
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
                action, _ = agent.predict(obs, deterministic=True)  # c
                observation, reward, done, result_sting, _ = environment.step(action)
                obs = observation

            except KeyboardInterrupt:
                raise


if __name__ == "__main__":
    test = Rl_test()
    test.test_gym_with_RL()
