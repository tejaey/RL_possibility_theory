import json
from gymnasium.spaces import discrete
from torch.nn import GaussianNLLLoss
from qnets import EnsembleDQN, SimpleDQN
from loss_func import td_loss_meta
from loss_func import *
from action_selection import SelectActionMeta
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
    select_action: SelectActionMeta,
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
    env: gym.Env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(capacity=100000)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
    num_ensemble = 5
    configs = {
        "update_steps": 1,
        "action_dim": action_dim,
        "batch_size": 128,
        "episodes": 400,
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
        ensemble_action_majority_voting(action_dim),
        ensemble_action_weighted_sum(action_dim),
    ]
    action_selection_names = ["weighted_sum", "majority_voting"]

    combinations = []
    for loss_name, loss_func in zip(loss_names, ensemble_loss_funcs):
        for action_name, action_func in zip(
            action_selection_names, action_selection_funcs
        ):
            combo_name = f"{loss_name}_{action_name}"
            combinations.append((combo_name, loss_func, action_func))
    print(combinations)
    cartpole_data = {}
    for combination in combinations:
        res = []
        for i in range(1):
            try:
                replay_buffer = ReplayBuffer(capacity=100000)
                online_qnet = EnsembleDQN(
                    state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
                )
                target_qnet = EnsembleDQN(
                    state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
                )
                hard_target_update(online_qnet, target_qnet)
                optimizers = [
                    optim.Adam(qnet.parameters(), lr=1e-3) for qnet in online_qnet.qnets
                ]
                res.append(
                    training_loop_qn(
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
                )
            except:
                pass
        cartpole_data[combination[0]] = res
    with open("cartpole.json", "w") as f:
        f.write(json.dumps(cartpole_data))

    env: gym.Env = gym.make("LunarLander-v3")
    replay_buffer = ReplayBuffer(capacity=100000)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
    num_ensemble = 5
    configs = {
        "update_steps": 8,
        "action_dim": action_dim,
        "batch_size": 128,
        "episodes": 2000,
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

    combinations = []
    for loss_name, loss_func in zip(loss_names, ensemble_loss_funcs):
        for action_name, action_func in zip(
            action_selection_names, action_selection_funcs
        ):
            combo_name = f"{loss_name}_{action_name}"
            combinations.append((combo_name, loss_func, action_func))
    print(combinations)
    lunarlanderdata = {}
    for combination in combinations:
        res = []
        for i in range(3):
            print("Doing Combination", combination[0])
            try:
                replay_buffer = ReplayBuffer(capacity=100000)
                online_qnet = EnsembleDQN(
                    state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
                )
                target_qnet = EnsembleDQN(
                    state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
                )
                hard_target_update(online_qnet, target_qnet)
                optimizers = [
                    optim.Adam(qnet.parameters(), lr=1e-3) for qnet in online_qnet.qnets
                ]
                res.append(
                    training_loop_qn(
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
                )
            except:
                pass
        lunarlanderdata[combination[0]] = res
    with open("lunarlander.json", "w") as f:
        f.write(json.dumps(lunarlanderdata))

    env: gym.Env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(capacity=100000)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
    num_ensemble = 5
    configs = {
        "update_steps": 1,
        "action_dim": action_dim,
        "batch_size": 128,
        "episodes": 500,
        "eps": 0.1,
        "tau_soft_update": 0.05,
    }
    GAMMA = 1
    td_loss_func = td_loss_ensemble(
        GAMMA=GAMMA,
        ALPHA=0.1,
        normalise=False,
        possibility_update="mle_max_update",
        use_ensemble_min=False,
    )
    select_action = ensemble_action_majority_voting(action_dim)
    #
    online_qnet = EnsembleDQN(
        state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
    )
    target_qnet = EnsembleDQN(
        state_dim, action_dim, num_ensemble=num_ensemble, hidden_dim=64
    )
    hard_target_update(online_qnet, target_qnet)
    optimizers = [optim.Adam(qnet.parameters(), lr=1e-3) for qnet in online_qnet.qnets]
    result = training_loop_qn(
        env=env,
        online_qnet=online_qnet,
        target_qnet=target_qnet,
        optimizers=optimizers,
        td_lossfunc=td_loss_func,
        select_action=select_action,
        replay_buffer=replay_buffer,
        configs=configs,
        print_info=True,
        discrete=True,
    )

    print(max(result))
#
#     target_qnet,
#     optimizers,
#     td_lossfunc: td_loss_meta,
#     select_action: SelectActionMeta,
#     replay_buffer: ReplayBuffer,
#     configs: dict,
#     print_info: bool,
#     discrete: bool,
# ) -> list[float]:
#     update_steps = configs.get("update_steps", 50)
#     action_dim = configs.get("action_dim", 1)
#     batch_size = configs.get("batch_size", 64)
#     episodes = configs.get("episodes", 500)
#     eps = configs.get("eps", 0.1)
#  pass
