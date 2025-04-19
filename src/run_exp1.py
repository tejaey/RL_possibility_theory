from experiments import maxmax_experiment
from utils import log_results
from multiprocessing import Pool


episodes = 20000
num_runs = 2


def process_env(env_name):
    game_results = maxmax_experiment(env_name, episodes=episodes, num_runs=num_runs)
    log_results(game_results, env_name)
    return game_results


if __name__ == "__main__":
    envs = ["HalfCheetah-v5", "Walker2d-v5", "Hopper-v5"]
    # Spawn as many workers as you have envs (or CPU cores)
    with Pool(processes=len(envs)) as pool:
        # This will distribute the env names across the pool
        results = pool.map(process_env, envs)
