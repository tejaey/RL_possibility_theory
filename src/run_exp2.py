from experiments import mean_var_experiment, ensemble_experiment
from utils import log_results

if __name__ == "__main__":
    num_runs = 2
    episodes = 10000
    for env_name in ["CartPole-v1", "LunarLander-v3"]:
        game_results = ensemble_experiment(
            env_name, episodes=episodes, num_ensemble=3, num_runs=num_runs
        )
        log_results(game_results, env_name + "_3_net")
        game_results = ensemble_experiment(
            env_name, episodes=episodes, num_ensemble=5, num_runs=num_runs
        )
        log_results(game_results, env_name + "_5_net")
        game_results = mean_var_experiment(
            env_name, episodes=episodes, num_runs=num_runs
        )
        log_results(game_results, env_name)
