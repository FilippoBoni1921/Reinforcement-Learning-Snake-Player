import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

def plots(title, x_label, y_label, file_name, data_dict,window_size,eval=False,base=True,str=0):

    plt.figure(figsize=(10, 6))
    # Smoothing only for the first three lines
    smoothed_data_dict = {key: np.convolve(np.pad(data_array, (window_size - 1, 0), mode='constant', constant_values=0), np.ones(window_size)/window_size, mode='valid') 
                          if key != 'Baseline' else data_array
                          for key, data_array in data_dict.items()}

    x_values = np.arange(len(smoothed_data_dict["PPO"]))

    # Plot each line
    for key, smoothed_data_array in smoothed_data_dict.items():
        if base==True:
            if key == 'Baseline':
                baseline_value = data_dict['Baseline']
                baseline_array = np.full(len(x_values), baseline_value)
                plt.plot(x_values[str:], baseline_array[str:], 'r--', label='Baseline')
            else:
                plt.plot(x_values[str:], smoothed_data_array[str:], label=f'{key}')
        else:
            if key != 'Baseline':
                plt.plot(x_values[str:], smoothed_data_array[str:], label=f'{key}')

   

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=16)  # Adjust fontsize for ticks

    if eval == True:
        plt.xlim(0, 1000)
        ax.xaxis.set_major_locator(ticker.FixedLocator([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,1100]))
        
    # Customize the plot
    plt.title(title, fontstyle="italic", fontsize=22)
    plt.xlabel(x_label, fontstyle="italic", fontsize=18)
    plt.ylabel(y_label, fontstyle="italic", fontsize=18)

    legend = plt.legend(fontsize=16)  # Adjust fontsize for legend
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.grid(True, color="black")

    plt.gca().spines['bottom'].set_linewidth(2)  # Adjust linewidth for bottom spine
    plt.gca().spines['top'].set_linewidth(2)  # Adjust linewidth for top spine
    plt.gca().spines['left'].set_linewidth(2)  # Adjust linewidth for left spine
    plt.gca().spines['right'].set_linewidth(2)  # Adjust linewidth for right spine
    

    
    if eval == True:
        plt.savefig(f"{file_name}.png",bbox_inches='tight')
    else:
        save_path = os.path.join("plots", f"{file_name}.png")
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == '__main__':

    window_size = 50

    title = "Rewards Comparison"
    x_label = ""
    y_label = "Reward"
    file_name = "rewards"

    ppo = np.load("arrays/reward_history_ppo.npy")
    dqn = np.load("arrays/reward_history_dqn.npy")
    a2c = np.load("arrays/reward_history_a2c.npy")
    baseline = np.load("arrays/mean_reward_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline # Example of Baseline as a single value
    }

    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "rewards_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)


    title = "Average Length Comparison"
    x_label = ""
    y_label = "Length"
    file_name = "length"

    ppo = np.load("arrays/length_history_ppo.npy")
    dqn = np.load("arrays/length_history_dqn.npy")
    a2c = np.load("arrays/length_history_a2c.npy")
    baseline = np.load("arrays/mean_length_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline  # Example of Baseline as a single value
    }


    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "length_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)

    title = "Eaten Fruit Comparison"
    x_label = ""
    y_label = "N Fruit"
    file_name = "fruit"

    ppo = np.load("arrays/fruit_history_ppo.npy")
    dqn = np.load("arrays/fruit_history_dqn.npy")
    a2c = np.load("arrays/fruit_history_a2c.npy")
    baseline = np.load("arrays/mean_fruit_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline  # Example of Baseline as a single value
    }

    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "fruit_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)

    title = "Boards with eaten body Comparison"
    x_label = ""
    y_label = "N Boards"
    file_name = "eaten_b"

    ppo = np.load("arrays/eaten_b_ppo.npy")
    dqn = np.load("arrays/eaten_b_dqn.npy")
    a2c = np.load("arrays/eaten_b_a2c.npy")
    baseline = np.load("arrays/mean_eaten_b_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline  # Example of Baseline as a single value
    }


    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "eaten_b_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)

    title = "Boards with Wall Hit"
    x_label = ""
    y_label = "N Boards"
    file_name = "hit"

    ppo = np.load("arrays/hit_history_ppo.npy")
    dqn = np.load("arrays/hit_history_dqn.npy")
    a2c = np.load("arrays/hit_history_a2c.npy")
    baseline = np.load("arrays/mean_hit_history_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline  # Example of Baseline as a single value
    }

    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "hit_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)

    title = "Boards with Winning Game"
    x_label = ""
    y_label = "N Boards"
    file_name = "win"

    ppo = np.load("arrays/win_history_ppo.npy")
    dqn = np.load("arrays/win_history_dqn.npy")
    a2c = np.load("arrays/win_history_a2c.npy")
    baseline = np.load("arrays/mean_win_history_baseline.npy")

    # Sample data dictionary with Baseline as a single value
    data_dict = {
        'PPO': ppo,
        'DQN': dqn,
        'A2C': a2c,
        'Baseline': baseline  # Example of Baseline as a single value
    }


    plots(title, x_label, y_label, file_name, data_dict,window_size)

    file_name = "win_zoom"
    plots(title, x_label, y_label, file_name, data_dict,window_size=100,base=False,str=5000)