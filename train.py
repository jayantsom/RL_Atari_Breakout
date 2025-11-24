import ale_py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
import config
import model
import memory
import utils

def init_state(env):
    state, _ = env.reset()
    state = utils.preprocess_frame(state)
    return deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)

def update_state(state, next_frame):
    state.append(utils.preprocess_frame(next_frame))
    return np.stack(state, axis=0)

def optimize_model(policy_net, target_net, optimizer):
    if memory.memory_size() < config.BATCH_SIZE:
        return 0.0, 0.0
    
    batch = memory.sample_memory(config.BATCH_SIZE)
    state_batch = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.uint8).to(config.DEVICE)
    action_batch = torch.tensor([e[1] for e in batch], dtype=torch.long).to(config.DEVICE)
    reward_batch = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(config.DEVICE)
    next_state_batch = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.uint8).to(config.DEVICE)
    done_batch = torch.tensor([e[4] for e in batch], dtype=torch.bool).to(config.DEVICE)

    current_q_values = model.dqn_forward(policy_net, state_batch).gather(1, action_batch.unsqueeze(1))
    next_q_values = model.dqn_forward(target_net, next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (config.GAMMA * next_q_values * ~done_batch)
    
    loss = utils.huber_loss(current_q_values.squeeze(), expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    
    # SIMPLE GRADIENT CLIPPING ONLY
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    
    optimizer.step()
    
    avg_q = current_q_values.mean().item()
    return loss.item(), avg_q

def train():
    # Initialize logging
    log_file = open('training_logs.txt', 'w')
    log_file.write("Frame,Reward,Epsilon,Loss,QValue\n")
    
    env = gym.make(config.ENV_NAME, obs_type=config.OBS_TYPE, frameskip=config.FRAME_SKIP,
                   repeat_action_probability=config.REPEAT_ACTION_PROBABILITY)
    action_size = env.action_space.n
    
    policy_net = model.create_dqn(action_size)
    target_net = model.create_dqn(action_size)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.RMSprop(policy_net.parameters(), lr=config.LEARNING_RATE, 
                            alpha=0.95, eps=0.01)
    memory.init_memory(config.MEMORY_SIZE)
    
    frame_idx = 0
    state = init_state(env)
    episode_reward = 0
    episode_rewards = []
    loss_history = []
    q_history = []
    
    print("Starting training for 1M frames...")
    start_time = time.time()
    
    while frame_idx < config.TOTAL_FRAMES:
        epsilon = utils.get_epsilon(frame_idx, config)
        
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), dtype=torch.uint8).unsqueeze(0).to(config.DEVICE)
                action = model.dqn_forward(policy_net, state_tensor).max(1)[1].item()
        else:
            action = random.randint(0, action_size - 1)
        
        next_frame, reward, done, truncated, _ = env.step(action)
        next_state = update_state(state.copy(), next_frame)
        memory.push_memory(np.array(state), action, reward, np.array(next_state), done)
        
        episode_reward += reward
        
        if done or truncated:
            state = init_state(env)
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            state = deque(list(next_state), maxlen=config.STACK_FRAMES)
        
        frame_idx += 1
        
        # Training step - ONLY IF WE HAVE ENOUGH MEMORY
        if frame_idx > config.REPLAY_START_SIZE and frame_idx % 4 == 0:  # Train every 4 frames
            loss, avg_q = optimize_model(policy_net, target_net, optimizer)
            loss_history.append(loss)
            q_history.append(avg_q)
        
        # Update target network
        if frame_idx % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Logging and printing - Every 5% progress (50k frames)
        if frame_idx % 50000 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_loss = np.mean(loss_history[-100:]) if loss_history else 0
            avg_q_value = np.mean(q_history[-100:]) if q_history else 0
            
            # Write to log file
            log_file.write(f"{frame_idx},{avg_reward:.2f},{epsilon:.3f},{avg_loss:.4f},{avg_q_value:.3f}\n")
            log_file.flush()
            
            # Print progress every 5% (50k frames)
            progress = (frame_idx / config.TOTAL_FRAMES) * 100
            print(f"Progress: {progress:.1f}% | Frame: {frame_idx} | Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f} | Avg Q: {avg_q_value:.3f}")
    
    # Save final model
    torch.save(policy_net.state_dict(), 'trained_model.pth', weights_only=True)
    log_file.close()
    env.close()
    
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed! Total time: {total_time:.2f} hours")
    print("Model saved as 'trained_model.pth'")

if __name__ == "__main__":
    train()