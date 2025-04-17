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
from loss_func import td_loss_meta, actor_critic_loss, actor_critic_loss_maxmax
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
from config import DEVICE

from training_loop import training_loop_qn, training_loop_ac


def mean_var_experiment(
    env, episodes, batch_size=256, hidden_dim=128, num_runs=1, GAMMA=0.99
) -> dict:
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n

    loss_functions = []
    action_selection_methods = []

    for beta in [-0.5, -0.25, 0, 0.25, 0.5]:
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
    loss_functions.append(
        (distributional_qn_loss(GAMMA=GAMMA, method="Wasserstein"), "wasserstein_loss")
    )
    experimental_results = {}

    configs = {
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episodes": episodes,
        "gamma": GAMMA,
        "eps": 0.1,
        "tau_soft_update": 0.05,
    }
    for loss_function in loss_functions:
        for action_selection_method in action_selection_methods:
            name = f"{loss_function[1]}_{action_selection_method[1]}"
            results = []
            for exp_n in range(num_runs):
                online_network = MeanVarianceQNetwork(
                    state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
                ).to(DEVICE)
                target_network = MeanVarianceQNetwork(
                    state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
                ).to(DEVICE)
                hard_target_update(online_network, target_network)
                optimizers = optim.Adam(online_network.parameters(), lr=1e-3)
                replay_buffer = ReplayBuffer()
                try:
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
                    results.append(rewards)
                except Exception as e:
                    print(e)
            experimental_results[name] = results
    return experimental_results
