from gymnasium import make
from stable_baselines3 import PPO
import time

# Create the environment with render mode
env = make("CartPole-v1", render_mode="human")

# Load the trained model
model = PPO.load("models/cartpole_final")

# Initialize statistics
num_attempts = 0
max_duration = 0
total_rewards = []

# Run multiple episodes to see the agent's performance
num_episodes = 5
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    num_attempts += 1
    
    print(f"\nStarting episode {episode + 1}")
    print(f"Current statistics:")
    print(f"Number of attempts: {num_attempts}")
    print(f"Maximum duration achieved: {max_duration} timesteps")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        time.sleep(0.01)  # Slow down the visualization
    
    # Update maximum duration if this episode lasted longer
    if episode_reward > max_duration:
        max_duration = episode_reward
    
    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1} finished with reward: {episode_reward}")

# Print final statistics
print("\nFinal Statistics:")
print(f"Total number of attempts: {num_attempts}")
print(f"Maximum duration achieved: {max_duration} timesteps")
print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
print(f"Best episode reward: {max(total_rewards)}")
print(f"Worst episode reward: {min(total_rewards)}")

env.close() 