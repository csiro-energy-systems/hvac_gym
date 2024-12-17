import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\CSIRO_Newcastle_gym_W1_.xlsx"
data_source = pd.ExcelFile(file)
our_method = pd.read_excel(data_source, 'test_results')
head = 0
end = 720
indoor_our_approach = np.array(our_method['indoor_temp'][head:end]).astype(float)
indoor_real = np.array(our_method['real_indoor'][head:end]).astype(float)
all_conumsption_our_approach = np.array(our_method['cooling_usage'][head:end]).astype(float)
all_conumsption_real = np.array(our_method['real_cooling_usage'][head:end]).astype(float)
up_boundary = np.array(our_method['boundary_1'][head:end])
low_boundary = np.array(our_method['boundary_0'][head:end])
price = np.array(our_method['price'][head:end])
outdoor_temp = np.array(our_method['outdoor air'][head:end]).astype(float)
reward = np.array(pd.read_excel(data_source, 'CSIRO_Newcastle_gym')['reward'][head:86400]).astype(float)
thermal_kpi_our_list = np.array(our_method['thermal_kpi'][head:end]).astype(float)
thermal_kpi_our_approach = float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
energy_kpi_our_list = np.array(our_method['energy_kpi'][head:end]).astype(float)
energy_kpi_our_approach = float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])
cost_kpi_our_list = np.array(our_method['cost_kpi'][head:end]).astype(float)
cost_kpi_our_approach = float(cost_kpi_our_list[-1]) - float(cost_kpi_our_list[0])

thermal_kpi_real = np.array(our_method['real_thermal_kpi'][head:end]).astype(float)
thermal_kpi_real = float(thermal_kpi_real[-1]) - float(thermal_kpi_real[0])
energy_kpi_real = np.array(our_method['real_energy_kpi'][head:end]).astype(float)
energy_kpi_real = float(energy_kpi_real[-1]) - float(energy_kpi_real[0])
cost_kpi_real = np.array(our_method['real_cost_kpi'][head:end]).astype(float)
cost_kpi_real = float(cost_kpi_real[-1]) - float(cost_kpi_real[0])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(16, 18), constrained_layout=True)
# fig = plt.figure(figsize=(18, 17))
gs = gridspec.GridSpec(5, 3, figure=fig)
# fig.suptitle('Bestest air: Peak heat days',y=0.915)  # Set a title for the whole figure

# First row spanning all columns
ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
ax1.set_title('BUILDSim')
ax1.plot(indoor_our_approach, label='our method', color='plum', linewidth=2)
ax1.plot(indoor_real, label='Benchmark 4', color='orange', linewidth=2)
ax1.plot(up_boundary, ':', color='gray')
ax1.plot(low_boundary, ':', color='gray')
ax1.legend(loc="upper right")
ax1.set_ylabel('Temperature (°C)')

ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

# Second row spanning all columns
ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
ax2.set_title('Outdoor temperature')
ax2.plot(outdoor_temp, color='black', label='Outdoor air temperature')
ax2.set_ylabel('Temperature (°C)')
ax2.legend(loc="lower right")

ax2.set_xticklabels([])

# third row spanning all columns
ax3 = fig.add_subplot(gs[2, :])  # This adds a subplot that spans the second row entirely
ax3.set_title('Energy use and ToU')
# Plot the price on the right y-axis
ax3_right = ax3.twinx()
ax3_right.plot(price, color='black', label='ToU')
ax3_right.set_ylabel('Price (units)')

# Plot the energy consumption values on the left y-axis
ax3.plot(all_conumsption_our_approach, label='our method', color='plum', linewidth=2)
ax3.plot(all_conumsption_real, label='Benchmark 4', color='orange', linewidth=2)
ax3.set_ylabel('Power (kW)')

# Combine legends from both axes
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines + lines2, labels + labels2, loc="upper right")

specific_dates = ['Feb 21st', 'Feb 22nd', 'Feb 23rd', 'Feb 24th', 'Feb 25th']
index_dates = [i * (len(all_conumsption_our_approach) // (len(specific_dates) - 1)) for i in
               range(len(specific_dates))]

# Check if index_dates are within valid range
print(index_dates)  # Should all be <= len(all_conumsption_our_approach)
ax3.set_xticks(index_dates)
ax3.set_xticklabels(specific_dates)

ax4 = fig.add_subplot(gs[3, :])
batch_size = 144
num_batches2 = len(reward) // batch_size
batch_rewards2 = [
    np.sum(reward[i * batch_size:(i + 1) * batch_size])
    for i in range(num_batches2)
]
timesteps = np.arange(1, num_batches2 + 1) * batch_size
ma_window =140  # Adjust the window size for smoother curve
smoothed_rewards = pd.Series(batch_rewards2).rolling(window=ma_window).mean().to_numpy()
# Calculate standard deviation for confidence intervals (optional)
rolling_std = pd.Series(batch_rewards2).rolling(window=ma_window).std().to_numpy()

decay_factor = np.linspace(0.6, 0.6, num_batches2)
adjusted_std = rolling_std * decay_factor

ax4.plot(timesteps, smoothed_rewards, label='Cumulative reward of the proposed method', color='plum', linewidth=3)

# Add confidence interval (optional)
# ax4.fill_between(timesteps, smoothed_rewards - adjusted_std, smoothed_rewards + adjusted_std, color='cyan',
#                  alpha=0.2)
# ax4.plot(timesteps, batch_rewards)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Cumulative reward', fontsize=12)
ax4.legend(loc="lower right", fontsize=12)
ax4.set_title('Convergence plot of the proposed method', fontsize=12)

# Explicitly clear automatic date locators or formatters
ax4.xaxis.set_major_locator(plt.NullLocator())
ax4.xaxis.set_minor_locator(plt.NullLocator())

# Third row with three individual plots
titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
colors = ['plum', 'orange']
labels = ['our method', 'Benchmark 4']


datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
benchmarks = [[thermal_kpi_real],
              [energy_kpi_real],
              [cost_kpi_real]]

for i in range(3):
    ax = fig.add_subplot(gs[4, i])
    ax.set_title(titles[i])
    all_values = [datasets[i]] + benchmarks[i]
    rects = ax.bar(labels, all_values, color=colors, linewidth=2)
    ax.set_ylabel(titles[i])

    # Adding value annotations
    for rect, value in zip(rects, all_values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Adjust y-limits to accommodate text annotations
    max_height = max(all_values)
    ax.set_ylim(0, max_height + 0.1 * max_height)

fig.tight_layout()
fig.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\csiro_2.png')
plt.show()


def thermal(t, coo, hea):
    result = (max((max((t - coo), 0) + max((hea - t), 0)), 1))
    return result


c_our, c_rule, c_rl1, c_rl2 = 0, 0, 0, 0
for x in range(len(all_conumsption_our_approach) - 1):
    c_our += price[x] * all_conumsption_our_approach[x]
    c_rule += price[x] * all_conumsption_real[x]

FI_our = 1 - c_our / c_rule

print('orginial FI our:', FI_our)