import numpy as np
import pandas as pd


# Thermal Discomfort
def calculate_thermal_discomfort_kpi(T_z, T_z_coo, T_z_hea, K_tdis):
    """
    Calculate the Thermal Discomfort KPI as per Equation (4)

    T_z: function representing the temperature of zone z at time t
    T_z_coo: function representing the cooling setpoint temperature of zone z at time t
    T_z_hea: function representing the heating setpoint temperature of zone z at time t
    """
    K_tdis += max(T_z - T_z_coo, 0) + max(T_z_hea - T_z, 0) * (10 / 60)
    return K_tdis


def calculate_energy_kpi(cool, heat, K_energy):
    area=150
    K_energy += ((cool + heat) * (10 / 60))/area
    return K_energy

def calculate_cost_kpi(cool, heat, K_cost, price):
    area=150
    K_cost += ((cool + heat) * price * (10 / 60))/area
    return K_cost

import pandas as pd
import numpy as np

def _ToU_():
    results = np.array([0.18, 0.32, 0.59])
    periods = ['off-peak', 'shoulder', 'peak']
    price_dict = {period: value for period, value in zip(periods, results)}
    return price_dict