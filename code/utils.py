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
