<<<<<<< HEAD
# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.

import pickle
from datetime import datetime, timedelta
from typing import Any, Callable, SupportsFloat
=======
import pickle
from datetime import datetime, timedelta
from typing import Any, SupportsFloat
>>>>>>> temp-branch

import pandas as pd
import plotly.graph_objects as go
from gymnasium import Env
<<<<<<< HEAD
from gymnasium.spaces import Box
from loguru import logger
from overrides import overrides
from pandas import DataFrame, Series
=======
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from loguru import logger
from overrides import overrides
from pandas import DataFrame
>>>>>>> temp-branch
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from tqdm import tqdm

from hvac_gym.gym.hvac_agents import HVACAgent
from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf
<<<<<<< HEAD
from hvac_gym.utils.data_utils import unique
=======
>>>>>>> temp-branch
from hvac_gym.vis.vis_tools import figs_to_html

pd.set_option("display.max_rows", 15, "display.max_columns", 20, "display.width", 300, "display.precision", 3)


class HVACGym(Env[DataFrame, DataFrame]):
    """Gym environment for simulating a HVAC system."""

    state: DataFrame

<<<<<<< HEAD
    def __init__(self, site_config: HVACSiteConf, reward_function: Callable[[Series], float], sim_start_date: datetime | None = None) -> None:
        """Initializes the environment with the given configuration.
        :param site_config: The configuration of the HVAC system
        :param reward_function: The reward function to use in the simulation. This allows the called to customise the reward returned by the step()
        function.  Inputs are the observations from the environment, and the output is a single float reward value.
        :param sim_start_date: The start date for the simulation, or None to just start from the beginning of the dataset
        """
        self.title: str = "HVAC Gym"
        self.site_config = site_config
        self.reward_function = reward_function
=======
    def __init__(self, site_config: HVACSiteConf, sim_start_date: datetime | None = None) -> None:
        """Initializes the environment with the given configuration.
        :param site_config: The configuration of the HVAC system
        :param sim_start_date: The start date for the simulation, or None to just start from the beginning of the dataset
        """
        self.site_config = site_config
>>>>>>> temp-branch
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
<<<<<<< HEAD
        self.all_inputs = list(unique([item for sublist in inputs for item in sublist]))
=======
        self.all_inputs = list(pd.unique([item for sublist in inputs for item in sublist]))
>>>>>>> temp-branch

        self.setpoints = site_config.setpoints

        """ Initialise properties required by the gym environment: """

        # Actions are to increase or decrease all setpoints by a relative amount
        self.action_space = Box(low=-100, high=100, shape=(len(setpoints),))

        # Observations: a 1d array of real numbers with lower and upper bounds describing the non-setpoint sensor readings
        self.observation_space = Box(low=-1000, high=10000, shape=(len(self.all_inputs),))

        self.reset()

    @overrides
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
<<<<<<< HEAD
    ) -> tuple[Any, dict[str, Any]]:
=======
    ) -> tuple[ObsType, dict[str, Any]]:
>>>>>>> temp-branch
        """Resets the environment to its initial state.

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # reset the simulation state
        self.state = Box(low=-1000, high=10000, shape=(len(self.all_inputs),))

        self.index = self.start_index

        return self.state, {}

    @overrides
<<<<<<< HEAD
    def step(self, action: DataFrame) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
=======
    def step(self, action: DataFrame) -> tuple[ObsType, SupportsFloat, bool, dict[Any, Any]]:
>>>>>>> temp-branch
        """Takes a step in the environment with the given action.
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
<<<<<<< HEAD
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
=======
>>>>>>> temp-branch
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

<<<<<<< HEAD
        if self.index > len(sim_df) - 1:
            raise ValueError(
                f"Reached end of simulation data at index {self.index}, timestamp: {current_time}. Please reset() the gym before calling step()."
            )

=======
>>>>>>> temp-branch
        # suppress settingswithcopy warning
        pd.options.mode.chained_assignment = None

        for model_conf in self.site_config.ahu_models:
            model = models[model_conf]
            predict_time = current_time + timedelta(minutes=model_conf.horizon_mins)

            inputs = model.feature_names_in_
            output = model_conf.output
            model_df = sim_df[inputs]

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

<<<<<<< HEAD
        self.state = sim_df.loc[current_time]
        obs = self.state
        reward = self.reward_function(self.state)
        terminated: bool = self.index >= len(sim_df) - 1
        info: dict[str, Any] = {}
        truncated = False
        return obs, reward, terminated, truncated, info
=======
        self.state = action
        obs = self.state
        reward = 0.0
        terminated: bool = False
        info: dict[Any, Any] = {}
        return obs, reward, terminated, info
>>>>>>> temp-branch

    @overrides
    def render(self, mode: str = "human") -> list[Figure]:
        """Renders the environment as two plotly subplots."""
        sim_df = self.sim_df
        idx = self.index
        time = sim_df.index[idx]
<<<<<<< HEAD
        inputs_no_lags = unique(self.all_inputs + self.setpoints)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Actual Data", "Simulated Data"))
=======
        inputs_no_lags = pd.unique(self.all_inputs + self.setpoints)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Simulated Data", "Actual Data"))
>>>>>>> temp-branch

        # plot the simulation results
        title = f"Simulation results for {self.site_config.site}"
        targets = [str(m.target) for m in self.site_config.ahu_models]
<<<<<<< HEAD
        targets_and_inputs = [c for c in list(unique(targets + [str(i) for i in inputs_no_lags])) if c in sim_df.columns]

        # plot the actuals for comparison
        p_actuals = self.actuals_df.query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_actuals.columns:
            fig.add_trace(go.Scatter(x=p_actuals.index, y=p_actuals[col], name=col, mode="lines", opacity=0.8), row=1, col=1)

        # plot simulated data
        p_sim = sim_df[targets_and_inputs].query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_sim.columns:
            fig.add_trace(go.Scatter(x=p_sim.index, y=p_sim[col], name=col, mode="lines", opacity=0.8), row=2, col=1)
=======
        p_sim = sim_df[list(set(targets + [str(i) for i in inputs_no_lags]))].query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_sim.columns:
            fig.add_trace(go.Scatter(x=p_sim.index, y=p_sim[col], name=col, mode="lines", opacity=0.8), row=1, col=1)

        # also plot the actuals for comparison
        p_actuals = self.actuals_df.query(f"'{self.sim_start_date}'<index and index<'{time}'").sort_index()
        for col in p_actuals.columns:
            fig.add_trace(go.Scatter(x=p_actuals.index, y=p_actuals[col], name=col, mode="lines", opacity=0.8), row=2, col=1)
>>>>>>> temp-branch

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
<<<<<<< HEAD
    env: HVACGym, agent: HVACAgent, site_config: HVACSiteConf, max_steps: int | None = None, show_plot: bool = False
) -> tuple[list[Any], list[SupportsFloat]]:
=======
    env: Env[DataFrame, DataFrame], agent: HVACAgent, site_config: HVACSiteConf, max_steps: int | None = None, show_plot: bool = False
) -> None:
>>>>>>> temp-branch
    """Convenience method that runs a simulation of the gym environment with the specified agent
    :param env: The gym environment to simulate
    :param agent: The HVACAgent insstance to use in the simulation
    :param max_steps: The maximum number of steps to simulate
    :param show_plot: Whether to show a plot of the results
    :param conf: The configuration of the gym environment
<<<<<<< HEAD
    :return: A tuple of the observations and rewards from the simulation
    """
    env.reset()
    last_observation = env.sim_df.loc[env.sim_df.index[0]]
=======
    """
    env.reset()
    last_observation = None
>>>>>>> temp-branch

    if max_steps is None:
        max_steps = len(env.sim_df)

<<<<<<< HEAD
    rewards = []
    observations = []
=======
>>>>>>> temp-branch
    step = 0
    for step in tqdm(range(max_steps), f"Running gym simulation with agent: {agent.name}"):
        try:
            env.title = agent.name
            t0 = datetime.now()
            action = agent.act(last_observation, step)
<<<<<<< HEAD
            observation, reward, done, trunc, infos = env.step(action)

            observations.append(observation)
            rewards.append(reward)

=======
            observation, reward, done, infos = env.step(action)
>>>>>>> temp-branch
            last_observation = observation
            logger.trace(f"Step {step} of {max_steps} done in {datetime.now() - t0}, observation: \n{observation}")

            # if specified, show a plot of the results incrementally while running
            if show_plot and (step % 6 == 0 or step == max_steps - 1):
                env.render()

<<<<<<< HEAD
            if done:
                logger.info(f"Reached end of simulation data at index {step}, timestamp: {env.sim_df.index[step]}.")
                break

        except KeyboardInterrupt:
            raise

    logger.info(f"Ended after {step+1} steps.")
=======
        except KeyboardInterrupt:
            raise

    logger.info(f"Ended after {step} steps.")
>>>>>>> temp-branch

    """ Always save the final plot to static html for reference """
    final_figs = env.render()
    title = f"Simulation results for {site_config.site}"
<<<<<<< HEAD
    figs_to_html(final_figs, f"output/{title}", show=show_plot)

    env.close()
    return observations, rewards
=======
    figs_to_html(final_figs, f"output/{title}", show=True)

    env.close()
>>>>>>> temp-branch


if __name__ == "__main__":
    pass
