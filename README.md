# AI Walker - Reinforcement Learning for Bipedal Walking

This project implements a reinforcement learning solution for the BipedalWalker-v3 environment using the PPO algorithm from Stable-Baselines3.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

To train the agent, run:
```bash
python train_walker.py
```

This will:
- Create a `models` directory
- Train the PPO agent for 1 million timesteps
- Save checkpoints every 10,000 timesteps
- Save the final model as `models/bipedal_walker_final`

### Visualizing the Trained Agent

To watch the trained agent walk, run:
```bash
python visualize_walker.py
```

This will:
- Load the trained model
- Run 5 episodes with visualization
- Show the reward for each episode

## Environment Details

The BipedalWalker-v3 environment simulates a two-legged walker that needs to learn to walk forward. The agent receives:
- 24-dimensional observation space (joint angles, velocities, etc.)
- 4-dimensional action space (joint torques)
- Reward based on forward progress and energy efficiency

## Hyperparameters

The PPO implementation uses the following hyperparameters:
- Learning rate: 3e-4
- Steps per update: 2048
- Batch size: 64
- Number of epochs: 10
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip range: 0.2 