import os
import torch
import multiprocessing as mp
from experiments import mean_var_experiment, ensemble_experiment
from utils import log_results
from config import DEVICE

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
