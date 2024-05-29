from abc import ABC, abstractmethod

from gymnasium import Env
from gymnasium.core import ObsType
from overrides import EnforceOverrides, overrides


class HVACAgent(ABC, EnforceOverrides):
    """Abstract class for defining a HVAC control agent."""

    def __init__(self, env: Env, name: str) -> None:
        """Initializes the agent with the given environment and name."""
        self.env = env
        self.name = name

    @abstractmethod
    def act(self, observations: ObsType, step: int) -> list[float]:
        """Returns an action for the given observation."""
        raise NotImplementedError


class HeatCoolAgent(HVACAgent):
    """Agent that just cycles between heating and cooling every timestep."""

    def __init__(self, env: Env, cycle_steps: int = 2) -> None:
        """Initializes the agent with the given environment and cycle steps."""
        super().__init__(env, f"Alternate Heating and Cooling every {cycle_steps} steps")
        self.cycle_steps = cycle_steps

    @overrides
    def act(self, observations: ObsType, step: int) -> list[float]:
        """Returns the action for the given observation and step."""
        if step % self.cycle_steps < self.cycle_steps / 2:
            return [100, 0, 100, 0]  # max cooling
        else:
            return [0, 100, 100, 0]  # max heating
