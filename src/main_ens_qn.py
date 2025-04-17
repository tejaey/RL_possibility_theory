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

from training_loop import training_loop_qn


## normal AC
if __name__ == "__main__":
    # Example: Visualize LunarLander performance
    env: gym.Env = gym.make("LunarLander-v3", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(capacity=100000)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
    num_ensemble = 5
    configs = {
        "update_steps": 8,
        "action_dim": action_dim,
        "batch_size": 128,
        "episodes": 100,
        "eps": 0.1,
        "tau_soft_update": 0.05,
    }
    GAMMA = 1
    ensemble_loss_funcs = []
    loss_names = []

    for update in ["mle", "no_update", "avg_likelihood_ema", "mle_max_update"]:
        loss_names.append(f"td_loss_ensemble_{update}_minTrue")
        ensemble_loss_funcs.append(
            td_loss_ensemble(
                GAMMA=GAMMA,
                ALPHA=0.1,
                normalise=False,
                possibility_update=update,
                use_ensemble_min=True,
            )
        )

    for update in ["mle", "avg_likelihood_ema", "mle_max_update"]:
        loss_names.append(f"td_loss_ensemble_{update}_minFalse")
        ensemble_loss_funcs.append(
            td_loss_ensemble(
                GAMMA=GAMMA,
                ALPHA=0.1,
                normalise=False,
                possibility_update=update,
                use_ensemble_min=False,
            )
        )

    loss_names.append("td_loss_ensemble_grad")
    ensemble_loss_funcs.append(
        td_loss_ensemble_grad(GAMMA=GAMMA, ALPHA=0.2, BETA=0.1, normalise=False)
    )

    action_selection_funcs = [
        ensemble_action_weighted_sum(action_dim),
        ensemble_action_majority_voting(action_dim),
    ]
    action_selection_names = ["weighted_sum", "majority_voting"]

    # Create combinations of loss and action selection functions
    combinations = []
    for loss_name, loss_func in zip(loss_names, ensemble_loss_funcs):
        for action_name, action_func in zip(
            action_selection_names, action_selection_funcs
        ):
            combo_name = f"{loss_name}_{action_name}"
            combinations.append((combo_name, loss_func, action_func))
    print("Combinations:", combinations)

    # Here we train on LunarLander with one of the combinations.
    # (For brevity, we select the first combination.)
    combination = select_combination(combinations)
    lunarlander_data = {}
    replay_buffer = ReplayBuffer(capacity=100000)
    online_qnet = EnsembleDQN(
        state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
    )
    target_qnet = EnsembleDQN(
        state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
    )
    hard_target_update(online_qnet, target_qnet)
    optimizers = [optim.Adam(qnet.parameters(), lr=1e-3) for qnet in online_qnet.qnets]
    rewards = training_loop_qn(
        env=env,
        online_qnet=online_qnet,
        target_qnet=target_qnet,
        optimizers=optimizers,
        td_lossfunc=combination[1],
        select_action=combination[2],
        replay_buffer=replay_buffer,
        configs=configs,
        print_info=True,
        discrete=True,
    )
    lunarlander_data[combination[0]] = rewards

    # Save the results to a json file.
    with open("lunarlander.json", "w") as f:
        f.write(json.dumps(lunarlander_data))

    # After training, run a visual demonstration of the LunarLander
    print("Visualizing LunarLander performance...")
    visualize_agent(env, online_qnet, combination[2], discrete=True)
