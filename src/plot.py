import json

from numpy import not_equal
from gymnasium.spaces import discrete
from torch.nn import GaussianNLLLoss
from qnets import EnsembleDQN, SimpleDQN
from loss_func import td_loss_meta
from loss_func import *
from action_selection import Qnet_SelectActionMeta
from action_selection import *
from utils import (
    ReplayBuffer,
    soft_update,
    hard_target_update,
    action_array,
    plot_improved_rewards,
)

import gymnasium as gym
from torch import optim
import logging


import logging

logging.basicConfig(
    level=logging.WARNING,  # Change to logging.TRACE with custom level if needed
    format="[{levelname}] {message}",
    style="{",
)


def training_loop_qn(
    env,
    online_qnet,
    target_qnet,
    optimizers,
    td_lossfunc: td_loss_meta,
    select_action: Qnet_SelectActionMeta,
    replay_buffer: ReplayBuffer,
    configs: dict,
    print_info: bool,
    discrete: bool,
) -> list[float]:
    update_steps = configs.get("update_steps", 50)
    action_dim = configs.get("action_dim", 1)
    batch_size = configs.get("batch_size", 64)
    episodes = configs.get("episodes", 500)
    eps = configs.get("eps", 0.1)
    rewards = []
    obs, _ = env.reset()
    # random experience to will replay buffer
    for _ in range(2 * batch_size):
        action = random.randint(0, action_dim - 1)
        if not discrete:
            action = action_array(action, action_dim)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(obs, action, reward, next_obs, done)
        logging.debug(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    # traning loop
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        count = 0
        while not done:
            count += 1
            action = select_action(state=obs, online_qnet=online_qnet, eps=eps)
            if not discrete:
                action = action_array(action=action, action_dim=action_dim)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            if count % update_steps == 0:
                batch = replay_buffer.sample(batch_size)
                loss = td_lossfunc(
                    batch=batch,
                    online_qnet=online_qnet,
                    target_qnet=target_qnet,
                    optimizers=optimizers,
                )
                logging.debug(f"Count :{count} Loss:{loss}")
                soft_update(
                    target=target_qnet,
                    online=online_qnet,
                    tau=configs.get("tau_soft_update", 0.05),
                )
        if print_info:
            print(f"Episode: {episode} | Rewards: {episode_reward}")
        rewards.append(episode_reward)
    return rewards


if __name__ == "__main__":
    import re
    import json
    import numpy as np

    def uses_min(string):
        return len(re.findall("minTrue", string)) >= 1

    def uses_majority_voting(string):
        return len(re.findall("voting", string)) >= 1

    def avg_results(resultsdict):
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
        for combo in resultsdict:
            game[combo] = avg2(rawgame[combo])[200:]

    def get_loss_function_name(combo_str):
        """
        Extracts the loss function portion of the combination name.
        For example, for "td_loss_ensemble_mle_minTrue_majority_voting", it returns
        "td_loss_ensemble_mle_minTrue".
        """
        if combo_str.split("_")[3] == "grad":
            return "grad"
        return re.search(r"(?<=td_loss_ensemble_).*?(?=_min)", combo_str).group(0)

    # Load raw data for CartPole.

    game_name = "Walker2d-v5"
    for game_name in [
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5_noisy",
        "Hopper-v5_noisy",
        "Walker2d-v5_noisy",
        "SparseHalfCheetah",
        "SparseWalker2DEnv",
        "SparseHopper",
    ]:
        with open(f"{game_name}.json", "r") as f:
            rawgame = json.loads(f.read())

        # Average over runs per combination.
        game = {}
        for combo in rawgame:
            game[combo] = avg2(rawgame[combo])[200:]

        # Plot overall data.
        plot_improved_rewards(game, smoothing_window=10, game=game_name)
    #
    # # Group by whether ensemble min is used.
    # use_min = []
    # not_use_min = []
    # for combo in rawgame:
    #     if uses_min(combo):
    #         use_min += rawgame[combo]
    #     else:
    #         not_use_min += rawgame[combo]
    # use_min_avg = avg2(use_min)
    # not_use_min_avg = avg2(not_use_min)
    # plot_improved_rewards(
    #     {"use_min": use_min_avg, "not_use_min": not_use_min_avg},
    #     game=game_name + " (minTrue vs others)",
    # )
    #
    # # Group by whether majority voting is used.
    # use_voting = []
    # not_use_voting = []
    # for combo in rawgame:
    #     if uses_majority_voting(combo):
    #         use_voting += rawgame[combo]
    #     else:
    #         not_use_voting += rawgame[combo]
    # use_voting_avg = avg2(use_voting)
    # not_use_voting_avg = avg2(not_use_voting)
    # plot_improved_rewards(
    #     {"use_voting": use_voting_avg, "not_use_voting": not_use_voting_avg},
    #     game=game_name + " (Majority Voting vs others)",
    # )
    #
    # # Group by loss functions.
    # loss_group = {}
    # for combo in rawgame:
    #     loss_name = get_loss_function_name(combo)
    #     if loss_name not in loss_group:
    #         loss_group[loss_name] = []
    #     loss_group[loss_name] += rawgame[combo]
    # loss_group_avg = {k: avg2(v) for k, v in loss_group.items()}
    # plot_improved_rewards(
    #     loss_group_avg,
    #     smoothing_window=10,
    #     game=game_name + " (Grouped by Update Method)",
    # )
