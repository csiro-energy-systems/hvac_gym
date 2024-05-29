import pickle
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from numpy import float32
from numpy.typing import NDArray
from overrides import overrides
from sklearn.base import RegressorMixin

from hvac_gym.sites.model_config import HVACModel, HVACSiteConf


class HVACGym(Env[NDArray[float32], NDArray[float32]]):
    """Gym environment for simulating a HVAC system."""

    state: NDArray[float32]

    def __init__(self, site_config: HVACSiteConf) -> None:
        """Initializes the environment with the given configuration."""
        self.site_config = site_config
        setpoints = site_config.setpoints

        site = site_config.site
        out_dir = site_config.out_dir

        self.models: dict[HVACModel, RegressorMixin] = {}
        # load models from disk
        for model_conf in site_config.ahu_models:
            with open(f"{out_dir}/{site}_{model_conf.target}_model.pkl", "rb") as f:
                self.models[model_conf] = pickle.load(f)

        self.sim_df = pd.read_parquet(f"{out_dir}/{site}_sim_df.parquet")

        inputs = [model_conf.inputs for model_conf in site_config.ahu_models]
        self.all_inputs = list(pd.unique([item for sublist in inputs for item in sublist]))

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
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to its initial state.

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        self.state = Box(low=-1000, high=10000, shape=(len(self.all_inputs),))
        return self.state, {}

    @overrides
    def step(self, action: NDArray[float32]) -> tuple[NDArray[float32], float, bool, dict[Any, Any]]:
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
        self.state = action
        obs = self.state
        reward = 0.0
        terminated: bool = False
        info: dict[Any, Any] = {}
        return obs, reward, terminated, info

    @overrides
    def render(self, mode: str = "human") -> None:
        """Renders the environment."""
        print(self.state)

    @overrides
    def close(self) -> None:
        """Closes the environment."""
        pass


if __name__ == "__main__":
    from hvac_gym.sites import newcastle_config

    site_config = newcastle_config.model_conf
    env = HVACGym(site_config)
    env.reset()
    env.step(np.array([0, 0, 0]))
    env.render()
    env.close()
