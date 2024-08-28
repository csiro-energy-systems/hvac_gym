# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.
from math import nan
from typing import Any

import pandas as pd
import pytest
from dch.paths.sem_paths import (
    ahu_chw_valve_sp,
    ahu_hw_valve_sp,
    ahu_oa_damper,
    ahu_sa_fan_speed,
    ahu_zone_temp,
)
from dch.paths.sem_paths import chiller_elec_power
from gymnasium import Env
from loguru import logger
from overrides import overrides
from pandas import DataFrame
from pandas import Series
from pendulum import parse, time

from hvac_gym.config.log_config import get_logger
from hvac_gym.gym.gym import HVACGym, run_gym_with_agent
from hvac_gym.gym.hvac_agents import HVACAgent
from hvac_gym.sites import newcastle_config
from hvac_gym.vis.vis_tools import figs_to_html
from hvac_gym.utils.thermal_comfort import comfort_range

logger = get_logger()


class FlexAgent(HVACAgent):
    """Agent that can be used to run repeated randomised simulations to quantify building power/comfort 'flexibility'"""

    def __init__(self, env: Env[DataFrame, DataFrame], flex_start: time, flex_end: time, cool_chwv_setpoint: float = 100) -> None:
        """Initializes the agent with the given environment and cycle steps."""
        super().__init__(env, f"Flexibility Simulation")

        from simple_pid import PID

        self.pid = PID(Kp=10, setpoint=22, sample_time=60 * 10, output_limits=(0, 100), starting_output=50)

        self.flex_end = flex_end
        self.flex_start = flex_start
        self.cool_chwv_setpoint = cool_chwv_setpoint
        self.df_schema = DataFrame(
            {
                str(ahu_chw_valve_sp): 0,
                str(ahu_hw_valve_sp): 0,
                str(ahu_oa_damper): 0,
                str(ahu_sa_fan_speed): 50,
            },
            index=[0],
        )

    @overrides
    def act(self, observations: Series, step: int) -> DataFrame:
        """Simple PI controller to control the chilled water valve and fan speed to maintain zone temp within a PMV-based comfort band."""
        zone_rh = 50
        pmv_upper = 0.5
        pmv_lower = -0.5

        ts = observations.name
        comfort_upper, comfort_lower = comfort_range(ts, zone_rh, pmv_upper, pmv_lower)
        if comfort_upper is None or comfort_lower is None:
            raise ValueError(f"Comfort range could not be calcualted for {ts}")
        zone_temp = observations[str(ahu_zone_temp)] if ahu_zone_temp in observations else observations["zone_temp_actual"]

        # default setpoint is the middle of the comfort band, unless in the flex period, then the upper bound
        if self.flex_start <= ts.time() <= self.flex_end:
            setpoint = comfort_upper
        else:
            setpoint = (comfort_upper + comfort_lower) / 2

        self.pid.time_fn = lambda: ts.timestamp()  # tell the PID loop to use the dataframe time instead of the system clock
        self.pid.setpoint = zone_temp
        new_chw_valve_sp = self.pid(setpoint)
        error = zone_temp - setpoint
        logger.info(
            f"Timestamp: {ts}, Setpoint: {setpoint:.1f}, Zone temp: {zone_temp:.1f}, Temp Error: {error:.1f}, chwv_setpoint: "
            f"{new_chw_valve_sp:.1f}"
        )

        self.df_schema[str(ahu_chw_valve_sp)] = new_chw_valve_sp

        return self.df_schema


class TestFlex:
    def flex_run(self, env: HVACGym, flex_start: time, flex_end: time, chwv_sp: float, max_steps: int = 24 * 6, show_plot: bool = False) -> DataFrame:
        """
        Runs a simulation on a single day with the given parameters
        :return:
        """
        agent = FlexAgent(env, cool_chwv_setpoint=chwv_sp, flex_start=flex_start, flex_end=flex_end)
        agent.name = f"Flexibility Simulation {flex_start} to {flex_end}"
        obs, rewards = run_gym_with_agent(env, agent, env.site_config, max_steps=max_steps, show_plot=show_plot)
        obs_df = pd.concat(obs, axis=1).T
        return obs_df

    @pytest.mark.manual("Long run, manual launch only")
    @pytest.mark.integration
    def test_gym_flex(self) -> None:
        """Measures flexibility"""
        max_steps = 24 * 6
        chwv_sp = 100
        start = parse("2023-12-14 00:00", tz="Australia/Sydney")  # 37°C day
        site_config = newcastle_config.model_conf

        def null_reward_function(observations: Series) -> float:
            return -1  # not used here

        env = HVACGym(site_config, reward_function=null_reward_function, sim_start_date=start)

        results: list[dict[str, Any]] = []

        baseline_df = self.flex_run(env, flex_start=time(0), flex_end=time(0), chwv_sp=chwv_sp, max_steps=max_steps, show_plot=False)
        baseline_power = baseline_df[str(chiller_elec_power)].sum() / 6
        actual_power = baseline_df["chiller_elec_power_actual"].sum() / 6

        # sweep the time range
        start_hour = 6
        end_hour = 18
        for flex_start in range(start_hour, end_hour, 1):
            flex_end = flex_start + 2
            obs_df = self.flex_run(env, flex_start=time(flex_start), flex_end=time(flex_end), chwv_sp=chwv_sp, max_steps=max_steps, show_plot=False)
            results.append(
                {
                    "description": "Flex",
                    "flex_start_hour": flex_start,
                    "flex_end_hour": flex_end,
                    "total_energy_kwh": obs_df[str(chiller_elec_power)].sum() / 6,
                }
            )
            figures = env.render()
            # Draw dotted lines for upper/lower temps from env.state using plotly
            from plotly import graph_objects as go

            figures[0].add_trace(
                go.Scatter(
                    x=obs_df.index,
                    y=obs_df["comfort_upper"],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Comfort Upper",
                )
            )
            figs_to_html(figures, f"output/flex_{flex_start}_{flex_end}.html", show=True)
            break  # TODO remove

        results.append({"description": "Baseline", "flex_start_hour": nan, "flex_end_hour": nan, "total_energy_kwh": baseline_power / 6})
        results.append({"description": "Actual", "flex_start_hour": nan, "flex_end_hour": nan, "total_energy_kwh": actual_power / 6})

        logger.info(f"Results: \n{pd.DataFrame(results)}")


if __name__ == "__main__":
    TestFlex().test_gym_flex()
