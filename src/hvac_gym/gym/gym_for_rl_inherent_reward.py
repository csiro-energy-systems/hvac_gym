import pickle
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, SupportsFloat

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dch.paths.sem_paths import oa_temp, zone_temp
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from loguru import logger
from overrides import overrides
from pandas import DataFrame, Series
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from tqdm import tqdm

from hvac_gym.gym.hvac_agents import HVACAgent
from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf
from hvac_gym.vis.vis_tools import figs_to_html

initial = 0
reset_para = 0
result_learning = ""
pd.set_option("display.max_rows", 15, "display.max_columns", 20, "display.width", 300, "display.precision", 3)


class HVACGym(Env[DataFrame, DataFrame]):
    """Gym environment for simulating a HVAC system."""

    state: DataFrame

    def __init__(self, site_config: HVACSiteConf, reward_function: Callable[[Series], float], sim_start_date: datetime | None = None) -> None:
        """Initializes the environment with the given configuration.
        :param site_config: The configuration of the HVAC system
        :param reward_function: The reward function to use in the simulation. This allows the called to customise the reward returned by the step()
        function.  Inputs are the observations from the environment, and the output is a single float reward value.
        :param sim_start_date: The start date for the simulation, or None to just start from the beginning of the dataset
        """
        self.site_config = site_config
        self.reward_function = reward_function
        setpoints = site_config.setpoints

        site = site_config.site
        out_dir = site_config.out_dir

        self.models: dict[HVACModelConf, RegressorMixin] = {}
        # load models from disk
        for model_conf in site_config.ahu_models:
            with open(f"{out_dir}/{site}_{model_conf.output}_model.pkl", "rb") as f:
                self.models[model_conf] = pickle.load(f)

        # read the simulation dataset
        self.sim_df = pd.read_parquet(f"{out_dir}/{site}_sim_df.parquet")

        # find first index >= sim_start_data, or 0 if not specified
        self.sim_start_date = pd.to_datetime(sim_start_date) if sim_start_date else self.sim_df.index[0]
        self.start_index = self.sim_df.index.searchsorted(self.sim_start_date, side="left")

        # make copies of the sim_df, so we can plot the actuals for comparison
        self.actuals_df = self.sim_df.copy()

        inputs = [model_conf.inputs for model_conf in site_config.ahu_models]
        self.all_inputs = list(pd.unique([item for sublist in inputs for item in sublist]))

        self.setpoints = site_config.setpoints

        """ Initialise properties required by the gym environment: """

        # Actions are to increase or decrease all setpoints by a relative amount
        # self.action_space = Box(low=-100, high=100, shape=(len(setpoints),))
        # self.observation_space = Box(low=-1000, high=10000, shape=(len(self.all_inputs),))

        self.action_space = Box(low=0, high=100, shape=(3,))
        # self.action_space = Box(low=-100, high=100, shape=(len(setpoints),))
        self.observation_space = Box(low=0, high=100, shape=(4,), dtype=float)
        # self.observation_space = Box(low=0, high=100, shape=(len(self.all_inputs),))

        self.reset()

    @overrides
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, new_parameter: bool = False):
        #     def reset(
        #     self,
        #     *,
        #     seed: int | None = None,
        #     options: dict[str, Any] | None = None,
        #     new_parameter: bool = False
        # ) -> tuple[ObsType, dict[str, Any]]:

        global reset_para
        if (reset_para == 0) or (new_parameter == True):
            self.index = self.start_index
            initial_observation = [0, 0, 0, 0]  # Example initial state
            self.state = initial_observation  # Box(low=-1000, high=10000, shape=(len(self.all_inputs),))
            reset_para = 1
            return np.array(initial_observation), 0
        else:
            reset_para = 1
            initial_observation = [0, 0, 0, 0]  # Example initial state
            return np.array(initial_observation), 0

    @overrides
    def step(self, action: DataFrame) -> tuple[ObsType, SupportsFloat, bool, dict[Any, Any]]:
        """Takes a step in the environment with the given action.
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        sim_df = self.sim_df
        models = self.models
        idx = self.index
        current_time = sim_df.index[idx]
        self.index += 1

        # suppress settingswithcopy warning
        pd.options.mode.chained_assignment = None

        for model_conf in self.site_config.ahu_models:
            model = models[model_conf]
            predict_time = current_time + timedelta(minutes=model_conf.horizon_mins)

            inputs = model.feature_names_in_
            output = model_conf.output
            model_df = sim_df[inputs]
            ##### Baseline3 lib generates the actions in np.array format, so I need to revise it back like this:
            columns = ["ahu_chw_valve_sp", "ahu_hw_valve_sp", "ahu_sa_fan_speed"]
            # Check if 'action' already has the right columns and structure
            if not isinstance(action, pd.DataFrame) or not all(col in action.columns for col in columns):
                # If 'action' is not a DataFrame or doesn't have the right columns
                # Assuming 'action' is an ndarray (3,) as expected
                if isinstance(action, np.ndarray):
                    # Check if action has extra dimensions and correct it
                    if action.ndim > 2:
                        action = action.squeeze()  # This attempts to remove extraneous dimensions if possible
                    elif action.shape[0] == 1:
                        action = action[0]  # This takes the first (and only) row of the action array
                    # Create a DataFrame from the numpy array
                    action = pd.DataFrame([action], columns=columns)
                # Add the 'od' column with default value 0
                action["ahu_oa_damper"] = 0
            # set actions here, if applicable to this model

            # set actions here, if applicable to this model
            model_actions = action[[c for c in inputs if c in action.columns]]
            model_df.loc[current_time, model_actions.columns] = model_actions.to_numpy()

            # set actions on sim_df also, just so they are rendered as set by the agent.
            sim_df.loc[current_time, model_actions.columns] = model_actions.to_numpy()

            # TODO set lags here also

            # get the model's prediction for the next timestep.
            # note: inputs/lags have already been down-shifted, so we can just apply prediction to the inputs at the timestamp directly.
            # this prediction will _apply_ to timestamp + horizon_mins for the model though
            predict_df = model_df.loc[[current_time]]

            # Run the prediction and update the building_df with result
            prediction = model.predict(predict_df)
            sim_df.loc[predict_time, str(output)] = prediction

        current_hour = current_time.hour
        self.state = sim_df.loc[current_time]
        self.state["hour"] = current_hour

        outdoor_temp = self.state[str(oa_temp)]
        indoor_temp = self.state[str(zone_temp)]
        if 7 <= current_hour <= 20:
            low_boundary = 21
            up_boundary = 24
        else:
            low_boundary = 15
            up_boundary = 30
        names = ["indoor_temp", "outdoor_temp", "up_boundary", "low_boundary"]
        obs = pd.Series(np.array([indoor_temp, outdoor_temp, up_boundary, low_boundary]), index=names)
        reward = self.reward_function(self.state)
        terminated: bool = False
        info: dict[Any, Any] = {}
        ahu_chw_valve_sp = action["ahu_chw_valve_sp"].values
        ahu_hw_valve_sp = action["ahu_hw_valve_sp"].values
        ahu_sa_fan_speed = action["ahu_sa_fan_speed"].values
        global initial
        global result_learning
        result_sting = (
            str(indoor_temp)
            + ","
            + str(low_boundary)
            + ","
            + str(up_boundary)
            + ","
            + str(reward)
            + ","
            + str(outdoor_temp)
            + ","
            + str(ahu_chw_valve_sp[0])
            + ","
            + str(ahu_hw_valve_sp[0])
            + ","
            + str(ahu_sa_fan_speed[0])
            + ","
            + str(current_hour)
            + "\n"
        )
        result_sting.replace("[", "").replace("]", "")
        if initial == 0:
            now = datetime.now()
            current_time = now.strftime("%dth-%b-%H-%M")
            result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\New_Newcastle_site_gym_" + current_time + "_.csv"
            f_1 = open(result_learning, "w+")
            record = "indoor temp,boundary,boundary,reward,outdoor air,ahu_chw_valve_sp,ahu_hw_valve_sp,fan_speed,hour\n"
            f_1.write(record)
            f_1.close()
            initial = 1
        else:
            initial = 1
            f_1 = open(result_learning, "a+")
            record = result_sting
            f_1.write(record)
            f_1.close()

        return obs, reward, terminated, 0, info

    @overrides
    def render(self, mode: str = "human") -> list[Figure]:
        """Renders the environment as two plotly subplots."""
        sim_df = self.sim_df
        idx = self.index
        time = sim_df.index[idx]
        inputs_no_lags = pd.unique(self.all_inputs + self.setpoints)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Simulated Data", "Actual Data"))

        # plot the simulation results
        title = f"Simulation results for {self.site_config.site}"
        targets = [str(m.target) for m in self.site_config.ahu_models]
        p_sim = sim_df[list(set(targets + [str(i) for i in inputs_no_lags]))].query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_sim.columns:
            fig.add_trace(go.Scatter(x=p_sim.index, y=p_sim[col], name=col, mode="lines", opacity=0.8), row=1, col=1)

        # also plot the actuals for comparison
        p_actuals = self.actuals_df.query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_actuals.columns:
            fig.add_trace(go.Scatter(x=p_actuals.index, y=p_actuals[col], name=col, mode="lines", opacity=0.8), row=2, col=1)

        # separate legends for each subplot (see https://community.plotly.com/t/plotly-subplots-with-individual-legends/1754/25)
        for i, yaxis in enumerate(fig.select_yaxes(), 1):
            legend_name = f"legend{i}"
            fig.update_layout({legend_name: dict(y=yaxis.domain[1], yanchor="top")}, showlegend=True)
            fig.update_traces(row=i, legend=legend_name)

        fig.update_layout(
            title=title,
            modebar_add=[
                "v1hovermode",
                "toggleSpikelines",
                "drawline",
                "hoverClosestGl2d",
                "hoverCompareCartesian",
                "autoScale2d",
                "zoom2d",
                "pan2d",
                "resetScale2d",
                "toImage",
                "zoomIn2d",
                "zoomOut2d",
                "lasso2d",
                "select2d",
            ],
        )

        return [fig]

    @overrides
    def close(self) -> None:
        """Closes the environment."""
        pass  # nothing to close


def run_gym_with_agent(
    env: Env[DataFrame, DataFrame], agent: HVACAgent, site_config: HVACSiteConf, max_steps: int | None = None, show_plot: bool = False
) -> tuple[list[ObsType], list[float]]:
    """Convenience method that runs a simulation of the gym environment with the specified agent
    :param env: The gym environment to simulate
    :param agent: The HVACAgent insstance to use in the simulation
    :param max_steps: The maximum number of steps to simulate
    :param show_plot: Whether to show a plot of the results
    :param conf: The configuration of the gym environment
    :return: A tuple of the observations and rewards from the simulation
    """
    env.reset()
    last_observation = None

    if max_steps is None:
        max_steps = len(env.sim_df)

    rewards = []
    observations = []
    step = 0
    for step in tqdm(range(max_steps), f"Running gym simulation with agent: {agent.name}"):
        try:
            env.title = agent.name
            t0 = datetime.now()
            action = agent.act(last_observation, step)
            observation, reward, done, infos = env.step(action)

            observations.append(observation)
            rewards.append(reward)

            last_observation = observation
            logger.trace(f"Step {step} of {max_steps} done in {datetime.now() - t0}, observation: \n{observation}")

            # if specified, show a plot of the results incrementally while running
            if show_plot and (step % 6 == 0 or step == max_steps - 1):
                env.render()

        except KeyboardInterrupt:
            raise

    logger.info(f"Ended after {step} steps.")

    """ Always save the final plot to static html for reference """
    final_figs = env.render()
    title = f"Simulation results for {site_config.site}"
    figs_to_html(final_figs, f"output/{title}", show=True)

    env.close()
    return observations, rewards


if __name__ == "__main__":
    pass
