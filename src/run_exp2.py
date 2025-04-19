from experiments import mean_var_experiment, ensemble_experiment
from utils import log_results
from multiprocessing import Pool

num_runs = 2
episodes = 10000


def process_task(task):
    env_name, kind, ensemble_size = task

    if kind == "ensemble":
        game_results = ensemble_experiment(
            env_name,
            episodes=episodes,
            num_ensemble=ensemble_size,
            num_runs=num_runs,
        )
        suffix = f"_{ensemble_size}_net"
    else:  # kind == 'mean_var'
        game_results = mean_var_experiment(
            env_name,
            episodes=episodes,
            num_runs=num_runs,
        )
        suffix = ""
    log_results(game_results, env_name + suffix)
    return env_name + suffix


if __name__ == "__main__":
    envs = ["CartPole-v1", "LunarLander-v3"]
    # build a list of all (env, experiment‑type, ensemble‑size) tasks
    tasks = []
    for env in envs:
        tasks.append((env, "ensemble", 3))
        tasks.append((env, "ensemble", 5))
        tasks.append((env, "mean_var", None))

    with Pool(processes=len(envs) * 2) as pool:  # adjust worker count as you like
        results = pool.map(process_task, tasks)

    print("Done:", results)
