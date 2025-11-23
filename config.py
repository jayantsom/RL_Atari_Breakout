import torch

# Environment
ENV_NAME = "ALE/Breakout-v5"
OBS_TYPE = "rgb"
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0.25

# Preprocessing
IMG_WIDTH = 84
IMG_HEIGHT = 84
STACK_FRAMES = 4

# Training - REDUCED FOR QUICK TEST
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.00025
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 1000  # Reduced from 10000
MEMORY_SIZE = 100000  # Reduced from 1000000
REPLAY_START_SIZE = 1000  # Reduced from 50000
TOTAL_FRAMES = 100000  # Reduced from 50000000 (100k instead of 50M)

# Exploration - Adjusted for shorter training
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 50000  # Reduced from 1000000