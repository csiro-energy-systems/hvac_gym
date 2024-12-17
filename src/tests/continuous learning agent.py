from pathlib import Path
import numpy as np
import pytest
import pytz
from dch.paths.sem_paths import ahu_chw_valve_sp, ahu_hw_valve_sp, ahu_oa_damper, ahu_sa_fan_speed, chiller_elec_power, zone_temp, oa_temp
from dch.utils.init_utils import cd_project_root
from pandas import DataFrame
from pendulum import parse
from pandas import DataFrame, Series
from hvac_gym.gym.Gym_for_RL_implementation_continuous import HVACGym, run_gym_with_agent  #Price ToU integarated
from hvac_gym.gym.performance_evaluation_metrics import calculate_thermal_discomfort_kpi,calculate_energy_kpi,_ToU_,calculate_cost_kpi
from hvac_gym.gym.hvac_agents import MinMaxCoolAgent
from hvac_gym.sites import newcastle_config
from datetime import datetime
from gymnasium.wrappers import NormalizeObservation
from gym.wrappers import RescaleAction
from tests. Training_testing_setup import training_setup, testing_setup
import math

cd_project_root()
Path("output").mkdir(exist_ok=True)


def optimized_outdoor_diff(x):
    beta = 1
    # delta = 17.5
    delta = 10
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
    T_mid = 17.5
    s = 5
    scale = 1.5
    return scale * np.tanh((T - T_mid) / s)

def nonlinear_temperature_response_2(T, up, low):
    T_mid = 17.5
    s = 10
    scale=int((up-low)/2)-1
    return scale* np.tanh((T - T_mid) / s)

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
    if (6 <= current_hour <= 7):
        load_shaping_mode = 1
    else:
        load_shaping_mode=0

    if (low_boundary < 20) and (up_boundary > 26):
        occupied_period = 0
    else:
        occupied_period = 1


    if (occupied_period == 0) and (load_shaping_mode == 0):
        edge = nonlinear_temperature_response_2(outdoor_temp, up_boundary, low_boundary)
        ref =  (up_boundary + low_boundary) / 2 + edge
        real_difference = abs(indoor_temp - ref)
        weight_T = 1 * (_ToU_()['off-peak'] - min_price) / (max_price - min_price)
        weight_E = 1 - weight_T
        consumption = cooling_usage #+ heating_usage
        if  consumption==0:
            thermal_reward = 1
            scenario = 1
            energy_reward = max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
        else:
            scenario = 1.1
            thermal_reward = penality_(real_difference)
            consumption = cooling_usage
            energy_reward = -max(0, ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

    elif (occupied_period == 0) and (load_shaping_mode == 1):  # day
        up_b_load_shaping = 26
        low_b_load_shaping = 20
        edge = nonlinear_temperature_response_2(outdoor_temp, up_b_load_shaping, low_b_load_shaping)
        ref = (up_boundary + low_boundary) / 2 + edge
        real_difference = abs(indoor_temp - ref)
        weight_T = 1 * (_ToU_()['shoulder'] - min_price) / (max_price - min_price)
        weight_E = 1 - weight_T
        if low_b_load_shaping <= indoor_temp <= up_b_load_shaping:
            scenario = 2
            thermal_reward = reward_(real_difference)
            consumption = cooling_usage
            energy_reward = max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            thermal_reward = penality_(real_difference)
            if low_b_load_shaping >= indoor_temp:
                scenario = 2.1
                consumption = cooling_usage
                energy_reward = -max(0, ((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 2.2
                consumption = cooling_usage
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
    elif occupied_period == 1:
        edge = nonlinear_temperature_response_2(outdoor_temp, up_boundary, low_boundary)
        ref = (up_boundary + low_boundary) / 2 + edge
        real_difference = abs(indoor_temp - ref)
        # weight_T = min((1 - normalized_price), normalized_price)
        weight_T = 1 * (_ToU_()['off-peak'] - min_price) / (max_price - min_price)
        weight_E = 1 - weight_T
        if low_boundary <= indoor_temp <= up_boundary:
            thermal_reward = reward_(real_difference)
            scenario = 3
            consumption = cooling_usage # + heating_usage     we dont have heating in Newcastle gym yet
            energy_reward = max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            thermal_reward = penality_(real_difference)
            if low_boundary >= indoor_temp:
                scenario = 3.1
                consumption = cooling_usage
                energy_reward = -max(0,((consumption - energy_min) / (energy_max - energy_min)))  # power is cooling

            else:
                # penalty case
                scenario = 3.2
                consumption = cooling_usage
                energy_reward = -max(0, 1 - ((consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
    return final_reward, thermal_reward, energy_reward, ref, weight_T, scenario
def the_one_under_test_outdoor (observations):
    indoor_temp = observations['indoor_temp']
    outdoor_temp = observations['outdoor_temp']
    up_boundary = observations['up_boundary']
    low_boundary = observations['low_boundary']
    cooling_usage = observations['cooling_usage']
    price = observations['price']
    current_hour = observations['current_hour']
    weight_edge = 0.0001
    # min_price = _ToU_()['off-peak'] - weight_edge
    # max_price = _ToU_()['peak'] + weight_edge
    # normalized_price = 1 * (price - min_price) / (max_price - min_price)
    # occupied_period = 0
    # load_shaping_mode = 0
    current_consumption_normalized = (cooling_usage - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 26
    real_bound_down = 20

    if (up_boundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =  (up_boundary + low_boundary) / 2
    else:
        wild_boundary = 0
        target = (up_boundary + low_boundary) / 2 # - 1 #

    outdoor_diff = abs(target - outdoor_temp)
    diff_curr = abs(indoor_temp - target)

    if (wild_boundary == 1):
        if (low_boundary < indoor_temp < up_boundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (real_bound_down < indoor_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        energy_weight = 1 - thermal_weight
        if (wild_boundary == 0):  # [21-24]
            thermal_reward = reward_(diff_curr)
            energy_reward = 1 - current_consumption_normalized
            final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
            scenario = 0

        else:  # [15-30]
            thermal_reward = reward_(diff_curr)
            energy_reward = 1 - current_consumption_normalized
            final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
            scenario = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        energy_weight = 1 - thermal_weight
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_temp):  # too low
                thermal_reward = penality_(diff_curr)
                energy_reward = current_consumption_normalized - 1
                final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
                scenario = 2
            elif (up_boundary < indoor_temp):  # too high
                thermal_reward = penality_(diff_curr)
                energy_reward = -current_consumption_normalized
                final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
                scenario = 3
            else:
                thermal_reward = penality_(diff_curr)
                final_reward = thermal_reward
                energy_reward = 0
                scenario = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_temp):  # too low
                thermal_reward = penality_(diff_curr)
                energy_reward = current_consumption_normalized - 1
                final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
                scenario = 5
            elif (real_bound_up < indoor_temp):  # too high
                thermal_reward = penality_(diff_curr)
                energy_reward = -current_consumption_normalized
                final_reward = thermal_weight * thermal_reward + (energy_weight) * energy_reward
                scenario = 6
            else:
                thermal_reward = penality_(diff_curr)
                final_reward = thermal_reward
                energy_reward = 0
                scenario = 7
    return final_reward, thermal_reward, energy_reward, target,thermal_weight, scenario


def reward_function (observations: Series) -> float:
    # reward, thermal_reward, energy_reward, ref, weight_T, scenario = reward_function_flexibility(observations) # the_one_under_test_outdoor

    reward, thermal_reward, energy_reward, ref, weight_T, scenario = the_one_under_test_outdoor(
        observations)  # the_one_under_test_outdoor

    return reward, thermal_reward, energy_reward, ref, weight_T, scenario

class Rl_test:
    now = datetime.now()
    current_time = now.strftime("%dth-%b-%H-%M")
    print(current_time)
    def train_gym_with_RL(self) -> None:
        time_interval = 10
        training_starts, training_days, episodes = training_setup()
        # testing_starts_1, testing_days_1,  testing_starts_2, testing_days_2, testing_starts_3, testing_days_3 = testing_setup()

        start = parse(training_starts).astimezone(pytz.timezone("Australia/Sydney"))
        print(f"training starts on: {start}")
        training_steps = int((60 * 24 / time_interval) * training_days)
        site_config = newcastle_config.model_conf

        _number_of_observations = 7
        environment = HVACGym(site_config, reward_function=reward_function, sim_start_date=start, amount_of_observations=_number_of_observations)
        environment.reset(new_parameter=True, sim_start_date=start)
        environment = NormalizeObservation(environment)
        print("Action Space:", environment.action_space)
        print("Observation Space:", environment.observation_space)

        from stable_baselines3 import DQN, A2C, PPO, SAC

        TRAIN_NEW_AGENT = True  # False #True
        MODEL_Name = "RL_agent_"
        if TRAIN_NEW_AGENT == True:
            _ent_coef_ = "auto"
            agent = SAC("MlpPolicy", environment, learning_rate=0.0001, batch_size=int(1 * (60 * 24 / time_interval)), ent_coef=_ent_coef_, gamma=0.9)
            for episode in range(episodes):
                print("This is the {}th episode".format(episode))
                environment.reset(new_parameter=True)
                agent.learn(total_timesteps=training_steps, reset_num_timesteps=False)
            agent.save(MODEL_Name)
            agent.save_replay_buffer('CSIRO_Newcastle_save_replay_buffer')
            print("The Learning progress is completed.")
        else:
            print("Please use the testing function to Launch the pre-trained agent.")

    def test_with_general_RL(self) -> None:
        print("We first test general RL.")
        time_interval = 10
        testing_starts, testing_days = testing_setup()
        # testing_starts = "2023-10-08 11:00:00+11:00"  # Must be within sim_df.index range

        start = parse(testing_starts).astimezone(pytz.timezone("Australia/Sydney"))
        print(f"Testing starts on: {start}")
        site_config_T = newcastle_config.model_conf
        _number_of_observations = 7
        environment = HVACGym(site_config_T, reward_function=reward_function, sim_start_date=start, amount_of_observations=_number_of_observations)
        environment.reset(new_parameter=True, sim_start_date=start)
        environment = NormalizeObservation(environment)
        from stable_baselines3 import DQN, A2C, PPO, SAC
        MODEL_Name = "RL_agent_"
        agent = SAC.load(MODEL_Name)
        print("Agent loaded.")
        agent.set_env(environment)
        test_steps = int(60 * 24 / time_interval * testing_days)
        obs = np.zeros(_number_of_observations)

        for step in range(test_steps):
            try:
                # print("now the current step is: ", step)
                action, _ = agent.predict(obs, deterministic=True)  # c
                observation, reward, done, result_sting, _ = environment.step(action)
                obs = observation

            except KeyboardInterrupt:
                raise
    def test_with_continuous_RL(self) -> None:
        print("We then test continuous RL.")
        time_interval = 10
        testing_starts, testing_days = testing_setup()
        start = parse(testing_starts).astimezone(pytz.timezone("Australia/Sydney"))
        site_config = newcastle_config.model_conf
        _number_of_observations = 7
        environment = HVACGym(site_config, reward_function=reward_function, sim_start_date=start, amount_of_observations=_number_of_observations)
        environment.reset(new_parameter=True)
        environment = NormalizeObservation(environment)
        from stable_baselines3 import DQN, A2C, PPO, SAC
        MODEL_Name = "RL_agent_"
        agent = SAC.load(MODEL_Name)
        print("Agent loaded.")
        # Reinitialize logger
        from stable_baselines3.common.logger import configure
        new_logger = configure(folder="./logs", format_strings=["stdout", "csv", "tensorboard"])
        agent._logger = new_logger

        agent.set_env(environment)
        test_steps = int(60 * 24 / time_interval * testing_days)
        obs = np.zeros(_number_of_observations)

        for step in range(test_steps):
            try:
                # print("now the current step is: ", step)
                action, _ = agent.predict(obs, deterministic=True)
                observation, reward, done, result_sting, info = environment.step(action)
                obs = observation
                fake_info = {"TimeLimit.truncated": False}
                agent.replay_buffer.add(obs, observation, action, reward, done, [fake_info])

                # Periodically re-train the agent
                if (step % (60 * 24 // time_interval)) == 0 and (step !=0):  # Train daily
                    print("Continuously training the agent during testing phase...")
                    agent.train(gradient_steps=1, batch_size=64)
            except KeyboardInterrupt:
                raise


if __name__ == "__main__":
    RL = Rl_test()
    import time
    start = time.time()
    RL.train_gym_with_RL()
    end = time.time()
    training_time=end - start
    print("Training_time is: ", training_time)
    print("Now testing is started.")
    start = time.time()
    RL.test_with_general_RL()
    end = time.time()
    test_time = end - start
    print("Testing_time is: ", test_time)
    start = time.time()
    RL.test_with_continuous_RL()
    end = time.time()
    test_time = end - start
    print("Testing_time is: ", test_time)
    print("Now the testing is completed.")
