from config import DEVICE
from experiments import possibilistic_model_exp
from utils import log_results

episodes = 1
num_runs = 1


def process_env(env_name):
    game_results = possibilistic_model_exp(
        env_name, episodes=episodes, num_runs=num_runs, method="zerostep"
    )
    log_results(game_results, env_name + "_zerostep")
    game_results = possibilistic_model_exp(
        env_name, episodes=episodes, num_runs=num_runs, method="onestep"
    )
    log_results(game_results, env_name + "_onestep")


if __name__ == "__main__":
    print(DEVICE)
    process_env("SparseWalker2DEnv")
    process_env("SparseHopper")
    process_env("Walker2d-v5")
    process_env("Hopper-v5")
