import os
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import time
import matplotlib.pyplot as plt
import numpy as np

# Create output directory for saved models
os.makedirs("models", exist_ok=True)

# Create and wrap the environment with render mode
env = make("CartPole-v1", render_mode="human")
env = DummyVecEnv([lambda: env])

# Custom callback to track attempts and duration
class StatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.num_attempts = 0
        self.max_duration = 0
        self.current_duration = 0
        self.last_print_time = time.time()
        self.dones = [False]  # Track episode completion
        self.durations = []  # Store all durations
        self.attempts = []   # Store attempt numbers

    def _on_step(self) -> bool:
        # Check if episode is done
        done = self.locals['dones'][0]
        if done:
            self.num_attempts += 1
            if self.current_duration > self.max_duration:
                self.max_duration = self.current_duration
            print(f"\nAttempt {self.num_attempts} finished with duration {self.current_duration}")
            # Store the data
            self.durations.append(self.current_duration)
            self.attempts.append(self.num_attempts)
            self.current_duration = 0
        else:
            self.current_duration += 1
        
        # Update stats every second
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            print(f"\rAttempts: {self.num_attempts}, Current Duration: {self.current_duration}, Max Duration: {self.max_duration}", end="")
            self.last_print_time = current_time
        return True

    def plot_durations(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.attempts, self.durations, 'b-', label='Duration')
        
        # Add moving average
        window_size = 10
        moving_avg = np.convolve(self.durations, np.ones(window_size)/window_size, mode='valid')
        plt.plot(self.attempts[window_size-1:], moving_avg, 'r-', label=f'{window_size}-episode moving average')
        
        plt.title('CartPole Training Progress')
        plt.xlabel('Attempt Number')
        plt.ylabel('Duration (timesteps)')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plt.savefig('training_progress.png')
        plt.close()

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Set up checkpointing to save the best model
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # Save more frequently
    save_path="./models/",
    name_prefix="cartpole"
)

# Create stats callback
stats_callback = StatsCallback()

# Train the model
total_timesteps = 20_000  # Increased timesteps for better training
model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, stats_callback],
    progress_bar=True
)

# Save the final model
model.save("models/cartpole_final")

# Close the environment
env.close()

# Print final statistics
print(f"\nTraining Statistics:")
print(f"Number of attempts: {stats_callback.num_attempts}")
print(f"Maximum duration achieved: {stats_callback.max_duration} timesteps")

# Create and save the plot
stats_callback.plot_durations()
print("\nTraining progress plot has been saved as 'training_progress.png'") 