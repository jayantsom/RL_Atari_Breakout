import numpy as np
import random

# Global memory variables
memory_buffer = []
memory_position = 0
memory_capacity = 0

def init_memory(capacity):
    global memory_buffer, memory_position, memory_capacity
    memory_buffer = [None] * capacity
    memory_position = 0
    memory_capacity = capacity

def push_memory(state, action, reward, next_state, done):
    global memory_buffer, memory_position
    memory_buffer[memory_position] = (state, action, reward, next_state, done)
    memory_position = (memory_position + 1) % memory_capacity

def sample_memory(batch_size):
    valid_samples = [x for x in memory_buffer if x is not None]
    if len(valid_samples) < batch_size:
        return valid_samples
    return random.sample(valid_samples, batch_size)

def memory_size():
    return len([x for x in memory_buffer if x is not None])