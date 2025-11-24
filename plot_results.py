import matplotlib.pyplot as plt
import numpy as np

def read_logs(log_file):
    frames, rewards, epsilons, losses, q_values = [], [], [], [], []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                frames.append(int(parts[0]))
                rewards.append(float(parts[1]))
                epsilons.append(float(parts[2]))
                losses.append(float(parts[3]))
                q_values.append(float(parts[4]))
    
    return frames, rewards, epsilons, losses, q_values

def plot_training(log_file):
    frames, rewards, epsilons, losses, q_values = read_logs(log_file)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(frames, rewards)
    plt.title('Average Rewards')
    plt.xlabel('Frames')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(frames, epsilons)
    plt.title('Epsilon')
    plt.xlabel('Frames')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(frames, losses)
    plt.title('Loss')
    plt.xlabel('Frames')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(frames, q_values)
    plt.title('Q-Values')
    plt.xlabel('Frames')
    plt.ylabel('Average Q-Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()

if __name__ == "__main__":
    plot_training('training_logs.txt')