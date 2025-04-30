import multiprocessing as mp
import os

import torch

from config import DEVICE
from experiments import ensemble_experiment, mean_var_experiment
from utils import log_results

num_runs = 1
episodes = 1

if __name__ == "__main__":
    for env_name in [
        "StochasticCartPole",
        "CartPole-v1",
        "SparseLunarLander",
        "LunarLander-v3",
    ]:
        ensemble_size = 5
        game_results = ensemble_experiment(
            env_name,
            episodes=episodes,
            num_ensemble=ensemble_size,
            num_runs=num_runs,
        )
        log_results(game_results, env_name + "_5ens")
