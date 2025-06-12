import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Function to choose an action based on probabilities
def choose_action(policy_net, state):
    state = torch.FloatTensor(state)
    probs = policy_net(state)
    action = np.random.choice(len(probs.detach().numpy()), p=probs.detach().numpy())
    return action, torch.log(probs[action])

# Training loop
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
gamma = 0.99  # Discount factor

for episode in range(500):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        action, log_prob = choose_action(policy_net, state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)
        state = new_state

    # Compute discounted rewards
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)

    discounted = torch.FloatTensor(discounted)
    log_probs = torch.stack(log_probs)
    loss = -torch.sum(log_probs * discounted)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")

env.close()

import os
from gym.wrappers.record_video import RecordVideo

# Create a new environment wrapped for video recording
video_env = RecordVideo(
    gym.make("CartPole-v1", render_mode="rgb_array"),
    video_folder="./videos",
    episode_trigger=lambda episode_id: True  # Always record
)

state, _ = video_env.reset()
done = False

while not done:
    action, _ = choose_action(policy_net, state)
    state, _, terminated, truncated, _ = video_env.step(action)
    done = terminated or truncated

video_env.close()

