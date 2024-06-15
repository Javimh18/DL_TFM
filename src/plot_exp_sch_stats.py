import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

RESULTS_DIR = '../results/exploration_schedule'
LIMIT1 = 12_000_000
LIMIT2 = 25_000_000

limit = LIMIT2
target_value = 1_000_000

# Load the data from the two log files
file_cnn_lin = '../checkpoints/ALE/MsPacman-v5/ddqn_cnn_agent/2024-05-31T20-48-14/log_cnn_lin'
file_cnn_exp = '../checkpoints/ALE/MsPacman-v5/ddqn_cnn_agent/2024-05-31T23-23-13/log_cnn_exp'
file_cnn_pow = '../checkpoints/ALE/MsPacman-v5/ddqn_cnn_agent/2024-06-05T13-08-15/log'

# create the results dir if necessary
os.makedirs(RESULTS_DIR, exist_ok=True)

# The files have different formats, so we need to handle them separately
# Read the log_cnn_pow_dqn.txt file
df_cnn_lin = pd.read_csv(file_cnn_lin, delim_whitespace=True, skiprows=1, names=[
    "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
    "MeanQValue", "TimeDelta", "Time"])

df_cnn_exp = pd.read_csv(file_cnn_exp, delim_whitespace=True, skiprows=1, names=[
    "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
    "MeanQValue", "TimeDelta", "Time"])

df_cnn_pow = pd.read_csv(file_cnn_pow, delim_whitespace=True, skiprows=1, names=[
    "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
    "MeanQValue", "TimeDelta", "Time"])

df_cnn_lin = df_cnn_lin[df_cnn_lin['Step']<limit]
df_cnn_exp = df_cnn_exp[df_cnn_exp['Step']<limit]
df_cnn_pow = df_cnn_pow[df_cnn_pow['Step']<limit]

# Plot the reward over time for both files
plt.figure(figsize=(14, 7))
plt.plot(df_cnn_lin['Step'], df_cnn_lin['MeanReward'], label='CNN Linear decay DDQN')
plt.plot(df_cnn_exp['Step'], df_cnn_exp['MeanReward'], label='CNN Exponential decay DDQN')
plt.plot(df_cnn_pow['Step'], df_cnn_pow['MeanReward'], label='CNN Product of Exponentials decay DDQN')
plt.axvline(x=1_000_000, color='red', linestyle='--', linewidth=2, label='Exploration rate fixed to 0.01 onwards')

plt.xlabel('Time Step')
plt.ylabel('Mean Reward')
plt.title('Mean Reward over Time for different schedules using CNN')
plt.legend()
plt.grid(True)
plt.savefig(f'{RESULTS_DIR}/exploration_rate_timesteps_limit_{limit}.png')

df_cnn_lin['CumulatimeTime'] = df_cnn_lin['TimeDelta'].cumsum()
df_cnn_lin['Difference'] = np.abs(df_cnn_lin['Step'] - target_value)
closest_index = df_cnn_lin['Difference'].idxmin()
cnn_lin_closest_row = df_cnn_lin.loc[closest_index]

df_cnn_exp['CumulatimeTime'] = df_cnn_exp['TimeDelta'].cumsum()
df_cnn_exp['Difference'] = np.abs(df_cnn_exp['Step'] - target_value)
closest_index = df_cnn_exp['Difference'].idxmin()
cnn_exp_closest_row = df_cnn_exp.loc[closest_index]

df_cnn_pow['CumulatimeTime'] = df_cnn_pow['TimeDelta'].cumsum()
df_cnn_pow['Difference'] = np.abs(df_cnn_pow['Step'] - target_value)
closest_index = df_cnn_pow['Difference'].idxmin()
cnn_pow_closest_row = df_cnn_pow.loc[closest_index]

# Plot the reward over time for both files
plt.figure(figsize=(14, 7))
plt.plot(df_cnn_lin['CumulatimeTime'], df_cnn_lin['MeanReward'], label='CNN Linear decay DDQN')
plt.plot(df_cnn_exp['CumulatimeTime'], df_cnn_exp['MeanReward'], label='CNN Exponential decay DDQN')
plt.plot(df_cnn_pow['CumulatimeTime'], df_cnn_pow['MeanReward'], label='CNN Product of Exponentials decay DDQN')
plt.axvline(x=cnn_lin_closest_row['CumulatimeTime'], color='purple', linestyle='--', linewidth=2, label='(LIN) Exploration rate fixed to 0.01 onwards')
plt.axvline(x=cnn_exp_closest_row['CumulatimeTime'], color='magenta', linestyle='--', linewidth=2, label='(EXP) Exploration rate fixed to 0.01 onwards')
plt.axvline(x=cnn_pow_closest_row['CumulatimeTime'], color='red', linestyle='--', linewidth=2, label='(POW) Exploration rate fixed to 0.01 onwards')
plt.xlabel('Time (s)')
plt.ylabel('Mean Reward')
plt.title('Mean Reward over Time for different schedules using CNN')
plt.legend()
plt.grid(True)
plt.savefig(f'{RESULTS_DIR}/exploration_rate_seconds_limit_{limit}.png')

print("Time of training for CNN Agent with POW Schedule: ", str(datetime.timedelta(seconds=df_cnn_pow['TimeDelta'].sum())))
print("Time of training for CNN Agent with EXP Schedule: ", str(datetime.timedelta(seconds=df_cnn_exp['TimeDelta'].sum())))
print("Time of training for CNN Agent with LIN Schedule: ", str(datetime.timedelta(seconds=df_cnn_lin['TimeDelta'].sum())))
