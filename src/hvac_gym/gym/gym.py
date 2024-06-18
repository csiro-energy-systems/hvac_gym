import pickle
from datetime import datetime, timedelta
from typing import Any, SupportsFloat

import pandas as pd
import plotly.express as px
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from loguru import logger
from overrides import overrides
from pandas import DataFrame
from plotly.graph_objs import Figure
from sklearn.base import RegressorMixin
from tqdm import tqdm

from hvac_gym.gym.hvac_agents import HVACAgent
from hvac_gym.sites.model_config import HVACModelConf, HVACSiteConf
from hvac_gym.vis.vis_tools import figs_to_html

pd.set_option("display.max_rows", 15, "display.max_columns", 20, "display.width", 300, "display.precision", 3)


class HVACGym(Env[DataFrame, DataFrame]):
    """Gym environment for simulating a HVAC system."""

    state: DataFrame

    def __init__(self, site_config: HVACSiteConf) -> None:
        """Initializes the environment with the given configuration."""
        self.site_config = site_config
        setpoints = site_config.setpoints

        site = site_config.site
        out_dir = site_config.out_dir

        self.models: dict[HVACModelConf, RegressorMixin] = {}
        # load models from disk
        for model_conf in site_config.ahu_models:
            with open(f"{out_dir}/{site}_{model_conf.output}_model.pkl", "rb") as f:
                self.models[model_conf] = pickle.load(f)

        self.sim_df = pd.read_parquet(f"{out_dir}/{site}_sim_df.parquet")

        inputs = [model_conf.inputs for model_conf in site_config.ahu_models]
        self.all_inputs = list(pd.unique([item for sublist in inputs for item in sublist]))

        self.setpoints = site_config.setpoints

        self.index = 0

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
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to its initial state.

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        self.state = Box(low=-1000, high=10000, shape=(len(self.all_inputs),))
        self.index = 0
        return self.state, {}

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

            # set actions here
            model_actions = action[[c for c in inputs if c in action.columns]]

            model_df.loc[current_time, model_actions.columns] = model_actions.to_numpy()

            # TODO set lags here also

            # get the model's prediction for the next timestep.
            # note: inputs/lags have already been down-shifted, so we can just apply prediction to the inputs at the timestamp directly.
            # this prediction will _apply_ to timestamp + horizon_mins for the model though
            predict_df = model_df.loc[[current_time]]

            # Run the prediction and update the building_df with result
            prediction = model.predict(predict_df)
            sim_df.loc[predict_time, str(output)] = prediction

        self.state = action
        obs = self.state
        reward = 0.0
        terminated: bool = False
        info: dict[Any, Any] = {}
        return obs, reward, terminated, info

    @overrides
    def render(self, mode: str = "human") -> Figure:
        """Renders the environment."""
        sim_df = self.sim_df
        idx = self.index
        time = sim_df.index[idx]
        inputs_no_lags = self.all_inputs

        print(self.state)

        # plot the simulation results
        title = f"Simulation results for {self.site_config.site}"
        targets = [str(m.target) for m in self.site_config.ahu_models]
        actuals = [c for c in sim_df.columns if c.endswith("_actual")]
        p = sim_df[list(set(targets + [str(i) for i in inputs_no_lags] + list(actuals)))].query(f"index<'{time}'").melt(ignore_index=False)

        fig = px.line(p, x=p.index, y="value", color="variable", title=title)
        return fig

    @overrides
    def close(self) -> None:
        """Closes the environment."""
        pass


def run_gym_with_agent(
    env: Env[DataFrame, DataFrame], agent: HVACAgent, site_config: HVACSiteConf, max_steps: int | None = None, show_plot: bool = False
) -> None:
    """Convenience method that runs a simulation of the gym environment with the specified agent
    :param env: The gym environment to simulate
    :param agent: The HVACAgent insstance to use in the simulation
    :param max_steps: The maximum number of steps to simulate
    :param show_plot: Whether to show a plot of the results
    :param conf: The configuration of the gym environment
    """
    env.reset()
    last_observation = None

    if max_steps is None:
        max_steps = len(env.sim_df)

    step = 0
    for step in tqdm(range(max_steps), f"Running gym simulation with agent: {agent.name}"):
        try:
            env.title = agent.name
            t0 = datetime.now()
            action = agent.act(last_observation, step)
            observation, reward, done, infos = env.step(action)
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
    figs_to_html([final_figs], f"output/{title}", show=True)

    env.close()


if __name__ == "__main__":
    pass
