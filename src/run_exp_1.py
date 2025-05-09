from experiments import mean_var_experiment
from utils import log_results

episodes = 1
num_runs = 1
if __name__ == "__main__":
    env_names = [
        "LunarLander-v3",
        "StochasticCartPole",
        "SparseLunarLander",
        "CartPole-v1",
    ]

    for env_name in env_names:
        game_results = mean_var_experiment(
            env_name,
            episodes=episodes,
            batch_size=256,
            hidden_dim=64,
            num_runs=num_runs,
        )
        log_results(game_results, env_name + "_meanvar")
