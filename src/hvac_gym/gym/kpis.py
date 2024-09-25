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


def calculate_energy_kpi(cool, heat, fan, K_energy):
    K_energy += (cool + heat + fan) * (10 / 60)
    return K_energy
