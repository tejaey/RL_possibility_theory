from experiments import maxmax_experiment
from utils import log_results

if __name__ == "__main__":
    episodes = 20000
    num_runs = 2
    for env_name in ["HalfCheetah-v5", "Walker2d-v5", "Hopper-v5"]:
        game_results = maxmax_experiment(env_name, episodes=episodes, num_runs=num_runs)
        log_results(game_results, env_name)
