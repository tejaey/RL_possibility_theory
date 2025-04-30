import json
from PIL.Image import Exif
from gymnasium.spaces import discrete
from torch.nn import GaussianNLLLoss
from torch.optim import optimizer
import action_selection
from qnets import (
    EnsembleCritic,
    EnsembleDQN,
    MeanVarianceQNetwork,
    SimpleDQN,
    Actor,
    SimpleCritic,
    QuantileModel,
    EnsembleQuantileModels,
)
from loss_func import (
    td_loss_meta,
    actor_critic_loss,
    actor_critic_loss_maxmax,
    td_loss_ensemble_grad_updated2,
    td_loss_ensemble,
    ActorCriticLossMaxMaxFix_onestep,
    ActorCriticLossMaxMaxFix_zerostep,
)
from loss_func import *
from action_selection import (
    Qnet_SelectActionMeta,
    mean_logvar_actionselection,
    mean_logvar_maxexpected,
    select_action_eps_greedy_meanvarQnet,
)
from action_selection import (
    AC_SelectAction,
    ensemble_action_majority_voting,
    ensemble_action_weighted_sum,
    single_dqn_eps_greedy,
)
from utils import (
    log_results,
    ReplayBuffer,
    soft_update,
    hard_target_update,
    action_array,
    plot_improved_rewards,
    visualize_agent,
    select_combination,
)
import gymnasium as gym
from torch import optim, utils
import logging
import matplotlib.pyplot as plt
from matplotlib import animation

from custom_env import SparseHalfCheetah, SparseWalker2DEnv, make_env
from config import DEVICE

from training_loop import training_loop_qn, training_loop_ac
from custom_env import make_env


def mean_var_experiment(
    env_name, episodes, batch_size=256, hidden_dim=128, num_runs=1, GAMMA=0.99
) -> dict:
    # env = gym.make(env_name)
    env = make_env(env_name)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n

    loss_functions = []
    action_selection_methods = []

    for beta in [-0.25, 0, 0.25, 0.5]:
        action_selection_methods.append(
            (
                mean_logvar_actionselection(action_dim=action_dim, beta=beta),
                f"log_uncertainity_{beta}",
            )
        )

    action_selection_methods.append(
        (mean_logvar_maxexpected(action_dim=action_dim), "maxexpected")
    )

    loss_functions.append(
        (distributional_qn_loss(GAMMA=GAMMA, method="Dkl"), "dkl_loss")
    )

    experimental_results = {}

    configs = {
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episodes": episodes,
        "batch_size": batch_size,
        "gamma": GAMMA,
        "eps": 0.1,
        "tau_soft_update": 0.05,
        "update_steps": 3,
    }
    for loss_function in loss_functions:
        for action_selection_method in action_selection_methods:
            name = f"{loss_function[1]}_{action_selection_method[1]}"
            results = []
            for exp_n in range(num_runs):
                print(name)
                online_network = MeanVarianceQNetwork(
                    state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
                ).to(DEVICE)
                target_network = MeanVarianceQNetwork(
                    state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
                ).to(DEVICE)
                hard_target_update(online_network, target_network)
                optimizers = optim.Adam(online_network.parameters(), lr=1e-2)
                replay_buffer = ReplayBuffer()
                try:
                    print(f"Run: {exp_n} | name {name}")
                    rewards = training_loop_qn(
                        env=env,
                        online_qnet=online_network,
                        target_qnet=target_network,
                        optimizers=optimizers,
                        td_lossfunc=loss_function[0],
                        select_action=action_selection_method[0],
                        replay_buffer=replay_buffer,
                        configs=configs,
                        print_info=True,
                        discrete=True,
                    )
                    print(max(rewards))
                    results.append(rewards)
                except Exception as e:
                    print(e)
            experimental_results[name] = results
    return experimental_results


def ensemble_experiment(
    env_name: str,
    episodes: int,
    num_ensemble: int = 5,
    hidden_dim: int = 64,
    num_runs: int = 1,
    GAMMA: float = 0.99,
    batch_size=128,
) -> dict:
    # Create environment
    env = make_env(env_name)
    # env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    loss_configs = []
    for upd in ["mle", "no_update", "mle_max_update"]:
        for use_min in [True, False]:
            name = f"td_ensemble_{upd}_min{use_min}"
            loss_fn = td_loss_ensemble(
                GAMMA=GAMMA,
                ALPHA=0.9,
                normalise=False,
                possibility_update=upd,
                use_ensemble_min=use_min,
            )
            loss_configs.append((loss_fn, name))

    action_methods = [
        (ensemble_action_weighted_sum(action_dim), "weighted_sum"),
        (ensemble_action_majority_voting(action_dim), "majority_vote"),
    ]

    results = {}

    configs = {
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episodes": episodes,
        "batch_size": batch_size,
        "eps": 0.1,
        "tau_soft_update": 0.05,
        "update_steps": 2,
    }

    t = 0
    for loss_fn, loss_name in loss_configs:
        for act_fn, act_name in action_methods:
            combo = f"{loss_name}_{act_name}"
            all_rewards = []

            for run in range(num_runs):
                t += 1
                print(f"Running {combo}, run {run + 1}/{num_runs}, number: {t}")

                # Instantiate ensembles
                online_qnet = EnsembleDQN(
                    state_dim,
                    action_dim,
                    num_ensemble=num_ensemble,
                    hidden_dim=hidden_dim,
                ).to(DEVICE)
                target_qnet = EnsembleDQN(
                    state_dim,
                    action_dim,
                    num_ensemble=num_ensemble,
                    hidden_dim=hidden_dim,
                ).to(DEVICE)
                hard_target_update(online_qnet, target_qnet)

                # One optimizer per member
                optimizers = [
                    optim.Adam(q.parameters(), lr=1e-3) for q in online_qnet.qnets
                ]

                buffer = ReplayBuffer(capacity=100_000)

                try:
                    rewards = training_loop_qn(
                        env=env,
                        online_qnet=online_qnet,
                        target_qnet=target_qnet,
                        optimizers=optimizers,
                        td_lossfunc=loss_fn,
                        select_action=act_fn,
                        replay_buffer=buffer,
                        configs=configs,
                        print_info=True,
                        discrete=True,
                    )
                    print(max(rewards))
                    all_rewards.append(rewards)
                except Exception as e:
                    print(e)

            results[combo] = all_rewards

    return results


def maxmax_experiment(
    env_name: str,
    episodes: int,
    num_runs: int = 2,
    GAMMA: float = 0.99,
    batch_size: int = 128,
    method: Literal["onestep", "zerostep"] = "onestep",
) -> dict:
    """
    Runs max-max actor-critic experiments over:
      - next_state_sample in ["null", "gaussian", "quantile"]
      - rollout_prob in [0.1, 0.5, 0.9] (ignored for "null")
    Each combination is run `num_runs` times.
    Uses 150 update steps/episode for "null" and 500 for the others.
    """
    results = {}

    for mode in ["null", "gaussian", "quantile"]:
        if mode == "null":
            probs = [None]  # no rollout_prob
        else:
            probs = [0.1]

        for p in probs:
            name = mode if p is None else f"{mode}_{p}"
            all_rewards = []

            for run in range(num_runs):
                print(f"Run [{run + 1}/{num_runs}] ─ mode={name}")

                env = make_env(env_name)
                buffer = ReplayBuffer(capacity=100_000)

                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]

                online_actor = Actor(state_dim, action_dim).to(DEVICE)
                target_actor = Actor(state_dim, action_dim).to(DEVICE)
                target_actor.load_state_dict(online_actor.state_dict())

                online_critic = SimpleCritic(state_dim, action_dim).to(DEVICE)
                target_critic = SimpleCritic(state_dim, action_dim).to(DEVICE)
                target_critic.load_state_dict(online_critic.state_dict())

                actor_opt = optim.Adam(online_actor.parameters(), lr=1e-3)
                critic_opt = optim.Adam(online_critic.parameters(), lr=1e-3)

                f = (
                    ActorCriticLossMaxMaxFix_zerostep
                    if method == "zerostep"
                    else ActorCriticLossMaxMaxFix_onestep
                )
                loss_fn = f(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    batch_size=batch_size,
                    gamma=GAMMA,
                    num_next_state_sample=5,
                    next_state_sample=mode,
                    rollout_depth=1 if mode != "null" else 0,
                    use_min=False,
                    rollout_prob=p or 0.0,
                )

                # 5) select-action fn
                select_action = AC_SelectAction(action_dim=action_dim)

                # 6) configs
                updates = 150 if mode == "null" else 200
                configs = {
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "episodes": episodes,
                    "batch_size": batch_size,
                    "eps": 0.1,
                    "tau_soft_update": 0.1,
                    "updates_per_episode": updates,
                }

                # 7) train
                try:
                    rewards = training_loop_ac(
                        env=env,
                        replay_buffer=buffer,
                        select_action_func=select_action,
                        loss_func=loss_fn,
                        online_actor=online_actor,
                        target_actor=target_actor,
                        actor_optimizer=actor_opt,
                        online_critic=online_critic,
                        target_critic=target_critic,
                        critic_optimizer=critic_opt,
                        configs=configs,
                        print_info=True,
                    )
                    print(max(rewards))
                    all_rewards.append(rewards)
                except Exception as e:
                    print(e)

                env.close()

            results[name] = all_rewards

    return results
