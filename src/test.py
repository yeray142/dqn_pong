import gymnasium as gym
import imageio
import argparse
import numpy as np
import stable_baselines3.common.atari_wrappers as atari_wrappers
import gymnasium.wrappers as ss

from stable_baselines3 import DQN

import ale_py
gym.register_envs(ale_py)


def load_model(model_path):
    """
    Loads the model given its path.
    
    Args:
        model_path: Path to the ZIP model file
    """
    model = DQN.load(model_path)
    print("Model '{}' loaded!".format(model_path))
    return model

def main():
    # Create argument parser for this script
    parser = argparse.ArgumentParser(description='Test DQN on Pong')
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the GYM environment')
    parser.add_argument('--model_path', type=str, default='pong_model.zip', help='Path to the model ZIP file')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to run')
    script_args, _ = parser.parse_known_args()
    
    # Load the model
    model = load_model(script_args.model_path)
    
    # Prepare for multiple episodes
    all_rewards = []
    best_reward = float('-inf')
    worst_reward = float('inf')
    best_frames = []
    worst_frames = []
    
    # Run multiple episodes
    for episode in range(script_args.num_episodes):
        print(f"Starting episode {episode+1}/{script_args.num_episodes}")
        
        # Load the environment (with render_mode for single environment approach)
        env = gym.make(script_args.env_name, render_mode="rgb_array")
        
        # Apply the same wrappers used during training - Using default parameters
        env = atari_wrappers.AtariWrapper(env)  # Use default parameters
        env = ss.FrameStackObservation(env, 2)  # Stack 2 frames as in training
        
        frames = []
        done = False
        total_reward = 0
        
        # Reset the environment
        obs, _ = env.reset()
        
        while not done:
            # Reshape the observation if needed
            if obs.shape == (2, 84, 84, 1):
                reshaped_obs = obs.squeeze(-1)
            else:
                reshaped_obs = obs
            
            # Use the reshaped observation for prediction
            action, _ = model.predict(reshaped_obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Check if done
            done = terminated or truncated
            
            # Capture frames
            frame = env.render()
            frames.append(frame)
        
        # Close environment
        env.close()
        
        # Store reward
        all_rewards.append(total_reward)
        print(f"Episode {episode+1} finished with total reward: {total_reward}")
        
        # Check if this is the best or worst episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames.copy()
            print(f"New best episode with reward: {best_reward}")
        
        if total_reward < worst_reward:
            worst_reward = total_reward
            worst_frames = frames.copy()
            print(f"New worst episode with reward: {worst_reward}")
    
    # Calculate statistics
    avg_reward = sum(all_rewards) / len(all_rewards)
    std_dev = (sum((r - avg_reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
    
    print(f"\nEvaluation Results:")
    print(f"Number of episodes: {script_args.num_episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"Best episode reward: {best_reward}")
    print(f"Worst episode reward: {worst_reward}")
    
    # Save the GIFs
    print("Saving best_episode.gif...")
    imageio.mimsave('best_episode.gif', best_frames, duration=0.05)
    
    print("Saving worst_episode.gif...")
    imageio.mimsave('worst_episode.gif', worst_frames, duration=0.05)
    
    print("Done!")

if __name__ == "__main__":
    main()