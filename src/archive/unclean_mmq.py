import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

device = "cpu"


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # States and actions are stored as numpy arrays.
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
        # Use tanh so that the action output is in [-1, 1]
        action = torch.tanh(self.fc3(x))
        return action


# Critic: Maps a (state, action) pair to a Q-value.
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


class QuantileModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QuantileModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, state_dim
        )  # output dimension = state dimension

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta = self.fc3(x)
        # Predicted next state is current state plus the learned delta.
        next_state_pred = state + delta
        return next_state_pred


def quantile_loss(pred, target, tau):
    error = target - pred
    loss = torch.max((tau - 1) * error, tau * error)
    return loss.mean()


def sample_candidates(lower, upper, num_samples):
    """
    Uniformly sample candidate next states from the interval [lower, upper].
    - lower, upper: Tensors of shape [batch_size, state_dim]
    - Returns: Tensor of shape [batch_size, num_samples, state_dim]
    """
    batch_size, state_dim = lower.shape
    uniform_samples = torch.rand(
        batch_size, num_samples, state_dim, device=lower.device
    )
    candidates = lower.unsqueeze(1) + (upper - lower).unsqueeze(1) * uniform_samples
    return candidates


# ---------------------------------------------------------
# 4. Initialize Environment and Networks
# ---------------------------------------------------------
env = gym.make("HalfCheetah-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# Two quantile networks: one for lower bound and one for upper bound.
quantile_lower = QuantileModel(state_dim, action_dim).to(device)
quantile_upper = QuantileModel(state_dim, action_dim).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
quantile_lower_optimizer = optim.Adam(quantile_lower.parameters(), lr=3e-4)
quantile_upper_optimizer = optim.Adam(quantile_upper.parameters(), lr=3e-4)

gamma = 0.99
batch_size = 256
num_candidate_samples = 10  # Number of candidates sampled from the learned interval
tau_target = 0.005  # Soft update rate for target networks

replay_buffer = ReplayBuffer(max_size=1000000)

# Initial random rollout to fill the replay buffer.
num_initial_rollouts = 10000
print("Collecting initial random experience...")
for _ in range(num_initial_rollouts):
    state, _ = env.reset()
    done = False

    action = env.action_space.sample()  # random action for exploration
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy()[0]
        action = action + np.random.normal(0, 0.5, size=action_dim)
        action = np.clip(action, -1, 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Store transition in replay buffer.
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        if done:
            break

    print(f"Episode {ep}: Reward = {episode_reward:.2f}")

    for _ in range(updates_per_episode): if replay_buffer.size() < batch_size:

        states_np, actions_np, rewards_np, next_states_np, dones_np = (
            replay_buffer.sample(batch_size)
        )
        states = torch.FloatTensor(states_np).to(device)
        actions = torch.FloatTensor(actions_np).to(device)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states_np).to(device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(device)

        pred_next_lower = quantile_lower(states, actions)
        pred_next_upper = quantile_upper(states, actions)
        loss_lower = quantile_loss(pred_next_lower, next_states, tau=0.5)
        loss_upper = quantile_loss(pred_next_upper, next_states, tau=0.5)

        quantile_lower_optimizer.zero_grad()
        loss_lower.backward()
        quantile_lower_optimizer.step()

        quantile_upper_optimizer.zero_grad()
        loss_upper.backward()
        quantile_upper_optimizer.step()

        candidate_next_states = sample_candidates(pred_next_lower, pred_next_upper, 10)
        bs, num_samples, _ = candidate_next_states.shape
        candidate_next_states_flat = candidate_next_states.reshape(
            bs * num_samples, state_dim
        )
        candidate_actions = target_actor(candidate_next_states_flat)
        candidate_qs = target_critic(candidate_next_states_flat, candidate_actions)
        candidate_qs = candidate_qs.reshape(bs, num_samples, 1)

        # Include the true observed next state as a candidate.
        true_next_state = next_states.unsqueeze(1)
        true_action = target_actor(next_states)
        true_q = target_critic(next_states, true_action).unsqueeze(1)
        all_candidate_qs = torch.cat([candidate_qs, true_q], dim=1)

        # Select the maximum Q-value (optimistic backup).
        max_q, _ = torch.max(all_candidate_qs, dim=1)
        target_q = rewards + gamma * (1 - dones) * max_q.detach()

        # ----- Update Critic -----
        current_q = critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(
                tau_target * param.data + (1 - tau_target) * target_param.data
            )
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(
                tau_target * param.data + (1 - tau_target) * target_param.data
            )


## try out the gaussian
## try out the like in the ensemble
## model based thing where you simulate the output - improve sample efficiency
## just use the rewards from the
##
