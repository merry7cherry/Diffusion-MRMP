from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from smd.runtime import runtime_from_env


def check_paths_ok(paths, obs_data, robot_radius: float = 0.05, threshold: float = 1e-3) -> bool:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("paths must have shape (R, T, 2)")

    num_robots, num_steps, _ = paths.shape
    for i in range(num_robots):
        for obstacle in obs_data:
            for t in range(num_steps):
                if (paths[i, t, 0] - obstacle[0][0]) ** 2 + (paths[i, t, 1] - obstacle[0][1]) ** 2 < (
                    robot_radius + obstacle[1]
                ) ** 2 - threshold:
                    return False

    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            for t in range(num_steps):
                if (paths[i, t, 0] - paths[j, t, 0]) ** 2 + (paths[i, t, 1] - paths[j, t, 1]) ** 2 < (
                    2.0 * robot_radius
                ) ** 2 - threshold:
                    return False
    return True


def evaluate_results(results_dir: str | Path) -> dict[str, float]:
    runtime = runtime_from_env()
    result_paths = sorted(Path(results_dir).rglob("paths.npy"))
    success_count = 0
    for path_file in result_paths:
        map_info = pickle.load(open(path_file.with_name("map_info.pkl"), "rb"))
        instance_file = runtime.instances_root / f"{map_info['map_name']}.pkl"
        instance_payload = pickle.load(open(instance_file, "rb"))
        map_data = instance_payload[map_info["instance_idx"]][2]
        obs_data = map_data[0]

        paths_data = np.load(path_file)
        num_agents = int(map_info["num_agents"])
        path_data = paths_data[0, :, : num_agents * 2].reshape(paths_data.shape[1], num_agents, 2).swapaxes(0, 1)
        if check_paths_ok(path_data, obs_data):
            success_count += 1

    return {
        "num_results": len(result_paths),
        "success_count": success_count,
        "success_rate": 0.0 if len(result_paths) == 0 else float(success_count) / float(len(result_paths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate collision feasibility for inference outputs.")
    parser.add_argument("--results-dir", default=str(runtime_from_env().inference_root))
    args = parser.parse_args()

    payload = evaluate_results(args.results_dir)
    print(payload)


if __name__ == "__main__":
    main()
