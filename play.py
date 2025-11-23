import ale_py
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import config
import model
import utils

def init_state(env):
    state, _ = env.reset()
    state = utils.preprocess_frame(state)
    return deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)

def update_state(state, next_frame):
    state.append(utils.preprocess_frame(next_frame))
    return np.stack(state, axis=0)

def play_game(model_path, episodes=5, render=True):
    env = gym.make(config.ENV_NAME, obs_type=config.OBS_TYPE, 
                   render_mode='human' if render else None)
    action_size = env.action_space.n
    
    policy_net = model.create_dqn(action_size)
    policy_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    policy_net.eval()
    
    for episode in range(episodes):
        state = init_state(env)
        total_reward = 0
        steps = 0
        
        while True:
            state_tensor = torch.tensor(np.array(state), dtype=torch.uint8).unsqueeze(0).to(config.DEVICE)
            
            with torch.no_grad():
                q_values = model.dqn_forward(policy_net, state_tensor)
                action = q_values.max(1)[1].item()
            
            next_frame, reward, done, truncated, _ = env.step(action)
            next_state = update_state(state.copy(), next_frame)
            
            total_reward += reward
            steps += 1
            state = deque(list(next_state), maxlen=config.STACK_FRAMES)
            
            if done or truncated:
                break
        
        print(f'Episode {episode + 1}: Score = {total_reward}, Steps = {steps}')
    
    env.close()

if __name__ == "__main__":
    play_game('trained_model.pth', episodes=3)