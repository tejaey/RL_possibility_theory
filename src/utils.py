import gymnasium as gym
import os
import json_tricks as json
from matplotlib import animation
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
import readchar


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
    if target is None:
        return
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )


def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def plot_improved_rewards(
    reward_dict, smoothing_window=10, game="GAME NAME", save: bool = False
):
    plt.figure(figsize=(12, 6))

    for algoname, reward_list in reward_dict.items():
        smoothed = np.convolve(
            reward_list, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        plt.plot(smoothed, label=algoname, linewidth=2)

    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Rewards", fontsize=14)
    plt.title(f"{game} Rewards Over Episodes", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f".images/{game}_rewards_plot_{current_time}.png"

    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved as: {filename}")

    plt.show()


def avg_results(rawresults, remove_first=0):
    def avg2(listoflist):
        # Computes an elementwise average across a list of lists.
        l = []
        i = 0
        while True:
            try:
                l.append(np.mean([lst[i] for lst in listoflist]))
                i += 1
            except Exception as e:
                print(e)
                break
        return l

    game = {}
    for combo in rawresults:
        game[combo] = avg2(rawresults[combo])[remove_first:]
    return game


def visualize_agent(env, online_qnet, select_action, discrete, delay=30):
    """
    Runs one episode using the given online_qnet and captures frames for visualization.
    Then creates and shows an animation.
    """
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action = select_action(state=obs, online_qnet=online_qnet, eps=0.0)
        if not discrete:
            action = action_array(action=action, action_dim=env.action_space.shape[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])
        return [patch]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=delay, blit=True
    )
    plt.show()


def select_combination(combinations):
    """
    Interactive selection menu for choosing a combination using arrow keys.
    """
    selected_index = 0  # Start at the first combination

    while True:
        print("\nSelect a Combination (Use ↑ ↓ keys and press Enter):\n")
        for i, (name, _, _) in enumerate(combinations):
            if i == selected_index:
                print(f"> {name}")  # Highlighted selection
            else:
                print(f"  {name}")

        key = readchar.readkey()

        if key == readchar.key.UP:
            selected_index = (selected_index - 1) % len(combinations)  # Move up
        elif key == readchar.key.DOWN:
            selected_index = (selected_index + 1) % len(combinations)  # Move down
        elif key == readchar.key.ENTER or key == readchar.key.SPACE:
            break  # Confirm selection

    return combinations[
        selected_index
    ]  # Return the selected combinatiorequirementrequirementn


def log_results(results_vals: dict, env_name_desc: str):
    log_dir = "./results_logs/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(log_dir, f"{env_name_desc}_{now}.json")

    with open(file_path, "w") as f:
        json.dump(results_vals, f)
