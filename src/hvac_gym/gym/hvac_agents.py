from abc import ABC, abstractmethod

from dch.paths.dch_paths import SemPath

# from dch.paths.sem_paths import (
#    ahu_chw_valve_sp,
#    ahu_hw_valve_sp,
#    ahu_oa_damper,
#    ahu_sa_fan_speed,
# )
from gymnasium import Env
from gymnasium.core import ObsType
from overrides import EnforceOverrides, overrides
from pandas import DataFrame

# FIXME currently override=False returns error
ahu_chw_valve_sp = SemPath(
    name="ahu_chw_valve_sp",
    path="AHU hasPart Chilled_Water_Coil feeds Chilled_Water_Valve hasPoint Valve_Position_Sensor",
    normalise_range=[0.0, 1.0],
    override=True,
)
ahu_hw_valve_sp = SemPath(
    name="ahu_hw_valve_sp",
    path="AHU hasPart Hot_Water_Coil feeds Hot_Water_Valve hasPoint Valve_Position_Sensor",
    normalise_range=[0.0, 1.0],
    override=True,
)
ahu_oa_damper = SemPath(
    name="ahu_oa_damper", path=["AHU hasPart Outside_Damper hasPoint Damper_Position_Sensor"], normalise_range=[0.0, 1.0], override=True
)
ahu_room_temp = SemPath(name="ahu_room_temp", path="AHU feeds HVAC_Zone hasPart Room hasPoint Air_Temperature_Sensor", override=True)
oa_temp = SemPath(
    name="oa_temp",
    path=["Equipment hasPoint Outside_Air_Temperature_Sensor", "Outside_Air_Temperature_Sensor"],
    desc="All outside air temperature sensors",
    override=True,
)
ahu_sa_fan_speed = SemPath(name="ahu_sa_fan_speed", path="AHU hasPart Supply_Fan hasPoint Speed_Sensor", override=True)


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
                str(ahu_sa_fan_speed): 0,
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


class MinMaxHotAgent(HVACAgent):
    """Agent that just cycles between heating and cooling every timestep."""

    def __init__(self, env: Env[DataFrame, DataFrame], cycle_steps: int = 2, hot_hwv_setpoint: float = 100) -> None:
        """Initializes the agent with the given environment and cycle steps."""
        super().__init__(env, f"Alternate Min-Max Cooling every {cycle_steps} steps")
        self.cycle_steps = cycle_steps
        self.hot_hwv_setpoint = hot_hwv_setpoint
        self.df_schema = DataFrame(
            {
                str(ahu_chw_valve_sp): 0,
                str(ahu_hw_valve_sp): 0,
                str(ahu_oa_damper): 0,
                str(ahu_sa_fan_speed): 0,
            },
            index=[0],
        )

    @overrides
    def act(self, observations: ObsType, step: int) -> DataFrame:
        """Returns the action for the given observation and step."""
        if step % self.cycle_steps < self.cycle_steps / 2:
            # max heating
            self.df_schema[str(ahu_hw_valve_sp)] = self.hot_hwv_setpoint
            return self.df_schema
        else:
            # no heating
            self.df_schema[str(ahu_hw_valve_sp)] = 0
            return self.df_schema
