# AI Walker - Reinforcement Learning for CartPole Balancing

This project implements a reinforcement learning solution for the CartPole-v1 environment using the PPO algorithm from Stable-Baselines3. The agent learns to balance a pole on a moving cart.

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
- Train the PPO agent for 20,000 timesteps
- Save checkpoints every 1,000 timesteps
- Save the final model as `models/cartpole_final`
- Generate a training progress plot (`training_progress.png`)
- Display real-time training statistics including:
  - Number of attempts
  - Current duration
  - Maximum duration achieved
  - Training metrics (loss, entropy, etc.)

### Visualizing the Trained Agent

To watch the trained agent balance the pole, run:
```bash
python visualize_walker.py
```

This will:
- Load the trained model
- Run 5 episodes with visualization
- Display statistics for each episode:
  - Current attempt number
  - Maximum duration achieved
  - Episode rewards
- Show final statistics including:
  - Total number of attempts
  - Maximum duration achieved
  - Average reward
  - Best and worst episode rewards

## Environment Details

The CartPole-v1 environment simulates a pole attached to a moving cart. The agent receives:
- 4-dimensional observation space (cart position, cart velocity, pole angle, pole angular velocity)
- 2-dimensional action space (push cart to the left or right)
- Reward of +1 for every timestep the pole remains balanced

## Hyperparameters

The PPO implementation uses the following hyperparameters:
- Learning rate: 3e-4
- Steps per update: 2048
- Batch size: 64
- Number of epochs: 10
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip range: 0.2

## Performance

The trained agent achieves:
- Maximum duration: 500 timesteps (10 seconds) - Perfect performance
- Average duration: 500 timesteps across all episodes
- Consistent performance: Achieves maximum duration in every episode
- Stable pole balancing with minimal oscillations
- Training statistics:
  - Total training time: ~7 minutes
  - Number of training attempts: 311
  - Best training duration: 499 timesteps
  - Final explained variance: 0.793
  - Final entropy loss: -0.567

## About

This project demonstrates the effectiveness of PPO in solving the CartPole balancing problem, achieving perfect performance through reinforcement learning. 