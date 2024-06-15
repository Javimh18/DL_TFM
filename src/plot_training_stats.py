import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

ROOT_DIR = '../checkpoints/ALE/'
ENV = 'MsPacman-v5'
RESULTS_DIR = '../results/training'

model_name_map = {
    "ddqn_cnn_agent": "CNN DDQN",
    "ddqn_swin_transformer_agent": "SWIN DDQN",
    "ddqn_vit_agent": "ViT DDQN"
}

if __name__ == "__main__":
    env_dir  = os.path.join(ROOT_DIR, ENV)
    models = os.listdir(env_dir)
    cnn_file = os.path.join(ROOT_DIR, ENV, "ddqn_cnn_agent", "2024-06-05T13-08-15", "log")
    swin_file = os.path.join(ROOT_DIR, ENV, "ddqn_swin_transformer_agent", "2024-06-08T22-20-02", "log")
    vit_file = os.path.join(ROOT_DIR, ENV, "ddqn_vit_agent", "2024-06-08T22-19-34", "log")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # read dataframes
    data_cnn_df = pd.read_csv(cnn_file, sep='\s+', skiprows=1, names=[
            "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
            "MeanQValue", "TimeDelta", "Time"])
    data_swin_df = pd.read_csv(swin_file, sep='\s+', skiprows=1, names=[
            "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
            "MeanQValue", "TimeDelta", "Time"])
    data_vit_df = pd.read_csv(vit_file, sep='\s+', skiprows=1, names=[
            "Episode", "Step", "Epsilon", "MeanReward", "MeanLength", "MeanLoss",
            "MeanQValue", "TimeDelta", "Time"])
    
    plt.figure(figsize=(14, 7))
    plt.plot(data_cnn_df['Step'], data_cnn_df['MeanReward'], label='CNN DDQN Agent')
    plt.plot(data_swin_df['Step'], data_swin_df['MeanReward'], label='SWIN DDQN Agent')
    plt.plot(data_vit_df['Step'], data_vit_df['MeanReward'], label='ViT DDQN Agent')
    # Plot a vertical red line at x = 1,000,000
    plt.axvline(x=1_000_000, color='red', linestyle='--', linewidth=2, label='Exploration rate fixed to 0.01 onwards')
        
    plt.xlabel('Time Step')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean Reward over Time-Step for DDQN using different Q-networks in {ENV} environment')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{RESULTS_DIR}/{ENV}_results_time_steps.png')
    
    target_value = 1_000_000
    data_cnn_df['CumulatimeTime'] = data_cnn_df['TimeDelta'].cumsum()
    data_cnn_df['Difference'] = np.abs(data_cnn_df['Step'] - target_value)
    closest_index = data_cnn_df['Difference'].idxmin()
    cnn_closest_row = data_cnn_df.loc[closest_index]
    
    data_swin_df['CumulatimeTime'] = data_swin_df['TimeDelta'].cumsum()
    data_swin_df['Difference'] = np.abs(data_cnn_df['Step'] - target_value)
    closest_index = data_swin_df['Difference'].idxmin()
    swin_closest_row = data_swin_df.loc[closest_index]
    
    data_vit_df['CumulatimeTime'] = data_vit_df['TimeDelta'].cumsum()
    data_vit_df['Difference'] = np.abs(data_cnn_df['Step'] - target_value)
    closest_index = data_vit_df['Difference'].idxmin()
    vit_closest_row = data_vit_df.loc[closest_index]
    # extracting the time in seconds where the agent stopped exploring
    vit_exp_rate_t = vit_closest_row['CumulatimeTime']
    swin_exp_rate_t = swin_closest_row['CumulatimeTime']
    cnn_exp_rate_t = cnn_closest_row['CumulatimeTime']

    # Plot the reward over time for both files
    plt.figure(figsize=(14, 7))
    plt.plot(data_cnn_df['CumulatimeTime'], data_cnn_df['MeanReward'], label='CNN DDQN Agent')
    plt.plot(data_swin_df['CumulatimeTime'], data_swin_df['MeanReward'], label='Swin Transformer DDQN Agent')
    plt.plot(data_vit_df['CumulatimeTime'], data_vit_df['MeanReward'], label='ViT DDQN Agent')
    
    plt.axvline(x=vit_exp_rate_t, color='purple', linestyle='--', linewidth=2, label='(ViT) Exploration rate fixed to 0.01 onwards')
    plt.axvline(x=swin_exp_rate_t, color='magenta', linestyle='--', linewidth=2, label='(SWIN) Exploration rate fixed to 0.01 onwards')
    plt.axvline(x=cnn_exp_rate_t, color='red', linestyle='--', linewidth=2, label='(CNN) Exploration rate fixed to 0.01 onwards')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean Reward over time for DDQN using different Q-networks in {ENV} environment')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{RESULTS_DIR}/{ENV}_results_time_seconds.png')
    
    print("Time of training for CNN DDQN Agent: ", str(datetime.timedelta(seconds=data_cnn_df['TimeDelta'].sum())))
    print("Time of training for Swim Transformer DDQN Agent: ", str(datetime.timedelta(seconds=data_swin_df['TimeDelta'].sum())))
    print("Time of training for ViT DDQN Agent: ", str(datetime.timedelta(seconds=data_vit_df['TimeDelta'].sum())))
        
    
    
