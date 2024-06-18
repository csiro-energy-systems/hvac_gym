from abc import ABC, abstractmethod

from dch.paths.sem_paths import (
    ahu_chw_valve_sp,
    ahu_hw_valve_sp,
    ahu_oa_damper,
)
from gymnasium import Env
from gymnasium.core import ObsType
from overrides import EnforceOverrides, overrides
from pandas import DataFrame


class HVACAgent(ABC, EnforceOverrides):
    """Abstract class for defining a HVAC control agent."""

    def __init__(self, env: Env[DataFrame, DataFrame], name: str) -> None:
        """Initializes the agent with the given environment and name."""
        self.env = env
        self.name = name

    @abstractmethod
    def act(self, observations: ObsType, step: int) -> DataFrame:
        """Returns an action for the given observation."""
        raise NotImplementedError


class MinMaxCoolAgent(HVACAgent):
    """Agent that just cycles between heating and cooling every timestep."""

    def __init__(self, env: Env[DataFrame, DataFrame], cycle_steps: int = 2, cool_chwv_setpoint: float = 100) -> None:
        """Initializes the agent with the given environment and cycle steps."""
        super().__init__(env, f"Alternate Min-Max Cooling every {cycle_steps} steps")
        self.cycle_steps = cycle_steps
        self.cool_chwv_setpoint = cool_chwv_setpoint
        self.df_schema = DataFrame(
            {
                str(ahu_chw_valve_sp): 0,
                str(ahu_hw_valve_sp): 0,
                str(ahu_oa_damper): 0,
            },
            index=[0],
        )

    @overrides
    def act(self, observations: ObsType, step: int) -> DataFrame:
        """Returns the action for the given observation and step."""
        if step % self.cycle_steps < self.cycle_steps / 2:
            # max cooling
            self.df_schema[str(ahu_chw_valve_sp)] = self.cool_chwv_setpoint
            return self.df_schema
        else:
            # no cooling
            self.df_schema[str(ahu_chw_valve_sp)] = 0
            return self.df_schema
