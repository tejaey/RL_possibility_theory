import json
from PIL.Image import Exif
from gymnasium.spaces import discrete
from torch.nn import GaussianNLLLoss
from qnets import (
    EnsembleCritic,
    EnsembleDQN,
    SimpleDQN,
    Actor,
    SimpleCritic,
    QuantileModel,
    EnsembleQuantileModels,
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


global device
device = "cpu"

logging.basicConfig(
    level=logging.WARNING,  # Change to a lower level if needed
    format="[{levelname}] {message}",
    style="{",
)


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


# max max approach
if __name__ == "__main__":
    iternumber = 2
    for env_name in [
        # "HalfCheetah-v5",
        # "Hopper-v5",
        # "Walker2d-v5",
        "SparseHalfCheetah",
        "SparseWalker2DEnv",
        "SparseHopper",
    ]:
        env: gym.Env = make_env(id=env_name)
        env_name += "noisy"
        # env: gym.Env = gym.make(env_name, render_mode="rgb_array")
        # env: gym.Env = gym.make(env_name, render_mode="rgb_array")
        experimental_results = {}

        next_state_sample_l = [
            "null",
            "gaussian",
            "triangular",
            "unif",
        ]
        for next_state_sample in next_state_sample_l:
            results = []
            for itern in range(iternumber):
                print(
                    f"Env: {env_name} | Sample Method: {next_state_sample} | Iter {itern}"
                )
                replay_buffer = ReplayBuffer(capacity=100000)

                state_dim: int = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]

                online_actor = Actor(state_dim, action_dim).to(device)
                online_critic = SimpleCritic(state_dim, action_dim).to(device)
                target_actor = Actor(state_dim, action_dim).to(device)
                target_critic = SimpleCritic(state_dim, action_dim).to(device)
                target_actor.load_state_dict(online_actor.state_dict())
                target_critic.load_state_dict(online_critic.state_dict())

                actor_optimizer = optim.Adam(online_actor.parameters(), lr=1e-3)
                critic_optimizer = optim.Adam(online_critic.parameters(), lr=1e-3)
                quantile_models = EnsembleQuantileModels(
                    state_dim=state_dim, action_dim=action_dim, quantiles=[0.05, 0.95]
                )
                quantile_optimizers = [
                    optim.Adam(model.parameters(), lr=1e-3)
                    for model in quantile_models.models
                ]

                select_action_func = AC_SelectAction(action_dim=action_dim)
                # loss_func = actor_critic_loss(
                #     state_dim=state_dim,
                #     action_dim=action_dim,
                #     batch_size=256,
                #     gamma=0.99,
                # )
                loss_func = actor_critic_loss_maxmax(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    batch_size=256,
                    gamma=0.99,
                    num_next_state_sample=5,
                    next_state_sample=next_state_sample,
                    use_min=False,
                )
                configs = {
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "episodes": 1000,
                    "eps": 0.1,
                    "tau_soft_update": 0.05,
                }
                try:
                    rewards = training_loop_ac(
                        env=env,
                        select_action_func=select_action_func,
                        loss_func=loss_func,
                        online_actor=online_actor,
                        target_actor=target_actor,
                        actor_optimizer=actor_optimizer,
                        online_critic=online_critic,
                        target_critic=target_critic,
                        critic_optimizer=critic_optimizer,
                        configs=configs,
                        print_info=True,
                        quantile_models=quantile_models,
                        quantile_optimizers=quantile_optimizers,
                    )
                    results.append(rewards)
                except Exception as e:
                    print(e)
            experimental_results[next_state_sample] = results
        with open(f"{env_name}.json", "w") as f:
            f.write(json.dumps(experimental_results))


# ensemble critic
if __name__ != "__main__":
    env: gym.Env = gym.make("Walker2d-v5", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(capacity=100000)

    state_dim: int = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    online_actor = Actor(state_dim, action_dim).to(device)
    online_critic = EnsembleCritic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim).to(device)
    target_critic = EnsembleCritic(state_dim, action_dim).to(device)

    target_actor.load_state_dict(online_actor.state_dict())
    target_critic.load_state_dict(online_critic.state_dict())

    actor_optimizer = optim.Adam(online_actor.parameters(), lr=1e-3)

    critic_optimizer = [
        optim.Adam(critic.parameters(), lr=1e-3) for critic in online_critic.critics
    ]

    select_action_func = AC_SelectAction(action_dim=action_dim)
    loss_func = actor_critic_loss_maxmax(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=256,
        gamma=0.99,
        num_next_state_sample=5,
        next_state_sample="gaussian",
        use_min=False,
    )
    configs = {
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episodes": 1000,
        "eps": 0.1,
        "tau_soft_update": 0.05,
    }
    training_loop_ac(
        env=env,
        select_action_func=select_action_func,
        loss_func=loss_func,
        online_actor=online_actor,
        target_actor=target_actor,
        actor_optimizer=actor_optimizer,
        online_critic=online_critic,
        target_critic=target_critic,
        critic_optimizer=critic_optimizer,
        configs=configs,
        print_info=True,
    )

if __name__ != "__main__":
    env: gym.Env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(capacity=100000)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    online_actor = Actor(state_dim, action_dim).to(device)
    online_critic = SimpleCritic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim).to(device)
    target_critic = SimpleCritic(state_dim, action_dim).to(device)
    target_actor.load_state_dict(online_actor.state_dict())
    target_critic.load_state_dict(online_critic.state_dict())

    actor_optimizer = optim.Adam(online_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(online_critic.parameters(), lr=1e-3)

    select_action_func = AC_SelectAction(action_dim=action_dim)
    loss_func = actor_critic_loss(batch_size=256, gamma=0.99)
    configs = {
        "update_steps": 8,
        "action_dim": action_dim,
        "episodes": 1000,
        "eps": 0.1,
        "tau_soft_update": 0.05,
    }
    training_loop_ac(
        env=env,
        select_action_func=select_action_func,
        loss_func=loss_func,
        online_actor=online_actor,
        target_actor=target_actor,
        actor_optimizer=actor_optimizer,
        online_critic=online_critic,
        target_critic=target_critic,
        critic_optimizer=critic_optimizer,
        configs=configs,
        print_info=True,
    )
## Make the rwards sparsers / stepwise

## Train a reward model

## Try in the tabular case, grid world

## Try model based approach with max max

## Try possibility ensemble quantile models in the first approach

## In the report

## possibility theory

## general RL explanation

## go to possibility function -
# gaussian it makes sense to sample from the gaussian
#
#
