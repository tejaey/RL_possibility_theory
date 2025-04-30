import random
import json
from torch.nn import GaussianNLLLoss
from qnets import (
    EnsembleCritic,
    EnsembleDQN,
    SimpleDQN,
    Actor,
    SimpleCritic,
    QuantileModel,
    EnsembleQuantileModels,
    MeanVarianceQNetwork,
)
from loss_func import td_loss_meta, actor_critic_loss, actor_critic_loss_maxmax
from loss_func import *
from action_selection import Qnet_SelectActionMeta
from action_selection import (
    AC_SelectAction,
    ensemble_action_majority_voting,
    ensemble_action_weighted_sum,
    single_dqn_eps_greedy,
)
from utils import (
    ReplayBuffer,
    soft_update,
    hard_target_update,
    action_array,
    plot_improved_rewards,
    visualize_agent,
    select_combination,
)
import gymnasium as gym
from torch import optim
import logging
import matplotlib.pyplot as plt
from matplotlib import animation

from custom_env import SparseHalfCheetah, SparseWalker2DEnv, make_env


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
    # Populate replay buffer with random experience
    for _ in range(2 * batch_size):
        action = random.randint(0, action_dim - 1)
        if not discrete:
            action = action_array(action, action_dim)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    # Training loop
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
            if count % update_steps == 1:
                batch = replay_buffer.sample(batch_size)

                loss = td_lossfunc(
                    batch=batch,
                    online_qnet=online_qnet,
                    target_qnet=target_qnet,
                    optimizers=optimizers,
                )
                # if print_info:
                #     print(f"Count :{count} Loss:{loss}")
                soft_update(
                    target=target_qnet,
                    online=online_qnet,
                    tau=configs.get("tau_soft_update", 0.05),
                )
        if print_info and episode % 20 == 0:
            print(f"Episode: {episode} | Rewards: {episode_reward} LOL")

        rewards.append(episode_reward)
    return rewards


# should use kwargs instead of configs maybe?
def training_loop_ac(
    env,
    select_action_func,
    loss_func,
    online_actor,
    target_actor,
    actor_optimizer,
    online_critic,
    target_critic,
    critic_optimizer,
    replay_buffer: ReplayBuffer,
    configs: dict,
    print_info: bool,
    quantile_models=None,
    quantile_optimizers=None,
):
    max_episode_steps = configs.get("max_episode_steps", 1000)
    updates_per_episode = configs.get("updates_per_episode", 50)

    action_dim = configs.get("action_dim", 1)
    batch_size = configs.get("batch_size", 64)
    episodes = configs.get("episodes", 500)

    state_dim = configs.get("state_dim")
    action_dim = configs.get("action_dim")
    for _ in range(batch_size):
        state, _ = env.reset()
        done = False
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(max_episode_steps):
            action = select_action_func(online_actor=online_actor, state=state, eps=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            loss = loss_func(
                replay_buffer=replay_buffer,
                online_actor=online_actor,
                target_actor=target_actor,
                actor_optimizer=actor_optimizer,
                online_critic=online_critic,
                target_critic=target_critic,
                critic_optimizer=critic_optimizer,
                quantile_models=quantile_models,
                quantile_optimizers=quantile_optimizers,
            )
        soft_update(
            target_critic,
            online_critic,
            tau=configs.get("tau_soft_update", 0.005),
        )
        soft_update(
            target_actor,
            online_actor,
            tau=configs.get("tau_soft_update", 0.005),
        )

        if print_info and ep % 10 == 0:
            print(f"Episode: {ep} | Reward: {episode_reward}")
        rewards.append(episode_reward)
    return rewards

    # quantile_optimizers=quantile_optimizers,
    # if ep == 200:
    #     loss_func.next_state_sample = "triangular"

    # for _ in range(updates_per_episode):
    #     break
    #     loss = loss_func(
    #         replay_buffer=replay_buffer,
    #         online_actor=online_actor,
    #         target_actor=target_actor,
    #         actor_optimizer=actor_optimizer,
    #         online_critic=online_critic,
    #         target_critic=target_critic,
    #         critic_optimizer=critic_optimizer,
    #     )
    #     soft_update(
    #         target_critic,
    #         online_critic,
    #         tau=configs.get("tau_soft_update", 0.005),
    #     )
    #     soft_update(
    #         target_actor,
    #         online_actor,
    #         tau=configs.get("tau_soft_update", 0.005),
    #     )
