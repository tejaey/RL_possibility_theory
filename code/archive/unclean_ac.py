import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh squashes the output to [-1, 1]
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


env = gym.make("HalfCheetah-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
gamma = 0.99
batch_size = 256
tau_target = 0.005  # Soft update rate for target networks

replay_buffer = ReplayBuffer(max_size=1000000)

# Collect initial random experience to fill the replay buffer.
num_initial_rollouts = 10000
print("Collecting initial random experience...")
for _ in range(num_initial_rollouts):
    state, _ = env.reset()
    done = False
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state

num_episodes = 1000
max_episode_steps = 1000  # Maximum steps per episode
updates_per_episode = 50  # Number of training updates after each episode

for ep in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(max_episode_steps):
        # Convert state to tensor and select action using the actor network.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy()[0]
        # Add Gaussian exploration noise.
        action = action + np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action, -1, 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            break

    print(f"Episode {ep}: Reward = {episode_reward:.2f}")

    # Perform several training updates after each episode.
    for _ in range(updates_per_episode):
        if replay_buffer.size() < batch_size:
            continue

        # Sample a mini-batch from the replay buffer.
        states_np, actions_np, rewards_np, next_states_np, dones_np = (
            replay_buffer.sample(batch_size)
        )
        states = torch.FloatTensor(states_np).to(device) actions = torch.FloatTensor(actions_np).to(device)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states_np).to(device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(device)

        # Compute the target Q-value using the target networks.
        next_actions = target_actor(next_states)
        next_q = target_critic(next_states, next_actions)
        target_q = rewards + gamma * (1 - dones) * next_q.detach()

        # Critic update: minimize the MSE between current Q and target Q.
        current_q = critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update: maximize the Q-value by minimizing the negative Q.
        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update of the target networks.
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(
                tau_target * param.data + (1 - tau_target) * target_param.data
            )
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(
                tau_target * param.data + (1 - tau_target) * target_param.data
            )

    # Save checkpoints periodically.
    if ep % 100 == 0:
        torch.save(actor.state_dict(), f"simple_actor_ep{ep}.pth")
        torch.save(critic.state_dict(), f"simple_critic_ep{ep}.pth")
