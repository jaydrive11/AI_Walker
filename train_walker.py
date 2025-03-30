import os
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import time

# Create output directory for saved models
os.makedirs("models", exist_ok=True)

# Create and wrap the environment with render mode
env = make("CartPole-v1", render_mode="human")
env = DummyVecEnv([lambda: env])

# Initialize counters
num_attempts = 0
max_duration = 0

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

# Train the model
total_timesteps = 10_000  # Reduced timesteps for faster training
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    progress_bar=True
)

# Save the final model
model.save("models/cartpole_final")

# Close the environment
env.close()

# Print final statistics
print(f"\nTraining Statistics:")
print(f"Number of attempts: {num_attempts}")
print(f"Maximum duration achieved: {max_duration} timesteps") 