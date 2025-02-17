import gymnasium as gym
import torch, random
from torch import optim
import numpy as np
import torch.nn as nn
from collections import deque
import typing
from typing import Callable, Any
import matplotlib.pyplot as plt
import math
import itertools
from datetime import datetime


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


def action_array(action, action_dim):
    x = np.zeros(action_dim)
    x[action] = 1
    return x


def soft_update(target, online, tau):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )


def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def plot_improved_rewards(reward_dict, smoothing_window=10, game="GAME NAME"):
    plt.figure(figsize=(12, 6))

    for algoname, reward_list in reward_dict.items():
        smoothed = np.convolve(
            reward_list, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        plt.plot(smoothed, label=algoname, linewidth=2)

    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Rewards", fontsize=14)
    plt.title(f"{game} Rewards Over Episodes (Smoothed)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{game}_rewards_plot_{current_time}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filename}")

    plt.show()


def training_loop_qn(
    env,
    online_qnet,
    target_qnet,
    optimizers,
    lossfunc,
    select_action,
    replay_buffer: ReplayBuffer,
    configs: dict,
    print_info: bool,
    discrete: bool,
) -> list[float]:
    update_steps = configs.get("update_steps", 50)
    action_dim = configs.get("action_dim", 1)
    batch_size = configs.get("batch_size", 64)
    episodes = configs.get("episodes", 500)
    rewards = []
    obs, _ = env.reset()
    # random experience to will replay buffer
    for _ in range(batch_size):
        action = random.randint(0, action_dim)
        if not discrete:
            action = action_array(action, action_dim)
        next_obs, reward, terminated, truncated, _ = env.step(
            action_array(action, action_dim=action_dim)
        )
        done = terminated or truncated
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()

    # traning loop
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        count = 0
        while not done:
            count += 1
            action = select_action(obs=obs, qnet=online_qnet, config=configs)
            if not discrete:
                action = action_array(action=action, action_dim=action_dim)
            next_obs, reward, terminated, truncated, info = env.step(
                action_array(action, action_dim)
            )
            done = terminated or truncated
            episode_reward += reward
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            if count % update_steps == 0:
                batch = replay_buffer.sample(1000)
                loss = lossfunc(
                    batch=batch,
                    online_qnet=online_qnet,
                    target_qnet=target_qnet,
                    optimizers=optimizers,
                )
                soft_update(
                    target=target_qnet,
                    online=online_qnet,
                    tau=configs.get("tau_soft_update"),
                )
        if print_info:
            print(f"Episode: {episode} | Rewards: {episode_reward}")
        rewards.append(episode_reward)
    return rewards
