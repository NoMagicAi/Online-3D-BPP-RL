import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(filename):
    """
    Loads training metrics from a CSV file and plots them.

    Args:
        filename (str): The path to the CSV file.
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return

    # Read the data into a pandas DataFrame directly from the CSV file
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Over Timesteps', fontsize=16)

    # Plot Space Ratio (a primary metric)
    axs[0, 0].plot(df['Timesteps'], df['space_ratio'], label='Space Ratio', color='blue')
    axs[0, 0].set_title('Space Ratio')
    axs[0, 0].set_xlabel('Timesteps')
    axs[0, 0].set_ylabel('Ratio')
    axs[0, 0].grid(True)

    # Plot Mean Reward
    axs[0, 1].plot(df['Timesteps'], df['reward_mean'], label='Mean Reward', color='green')
    axs[0, 1].set_title('Mean Reward (Last 10 Episodes)')
    axs[0, 1].set_xlabel('Timesteps')
    axs[0, 1].set_ylabel('Reward')
    axs[0, 1].grid(True)

    # Plot All Losses on one graph
    axs[1, 0].plot(df['Timesteps'], df['loss_entropy'], label='Entropy Loss', alpha=0.8)
    axs[1, 0].plot(df['Timesteps'], df['loss_value'], label='Value Loss', alpha=0.8)
    axs[1, 0].plot(df['Timesteps'], df['loss_action'], label='Action Loss', alpha=0.8)
    axs[1, 0].set_title('Losses')
    axs[1, 0].set_xlabel('Timesteps')
    axs[1, 0].set_ylabel('Loss Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Items Packed
    axs[1, 1].plot(df['Timesteps'], df['items_packed'], label='Items Packed', color='purple')
    axs[1, 1].set_title('Items Packed')
    axs[1, 1].set_xlabel('Timesteps')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].grid(True)

    # Plot FPS
    axs[2, 0].plot(df['Timesteps'], df['FPS'], label='FPS', color='red')
    axs[2, 0].set_title('Frames Per Second (FPS)')
    axs[2, 0].set_xlabel('Timesteps')
    axs[2, 0].set_ylabel('FPS')
    axs[2, 0].grid(True)

    # Plot PopArt Stats
    ax_popart = axs[2, 1]
    ax_popart.plot(df['Timesteps'], df['popart_mean'], label='PopArt Mean', color='orange')
    ax_popart.plot(df['Timesteps'], df['popart_std'], label='PopArt Std', color='cyan')
    ax_popart.set_title('PopArt Stats')
    ax_popart.set_xlabel('Timesteps')
    ax_popart.set_ylabel('Value')
    ax_popart.legend()
    ax_popart.grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # Prompt the user to enter the CSV file name
    csv_filename = input("Please enter the path to the CSV file: ")
    plot_training_metrics(csv_filename)