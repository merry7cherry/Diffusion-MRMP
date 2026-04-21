from __future__ import annotations

import argparse
import pickle
from datetime import datetime

from smd.common.experiments import MultiAgentPlanningExperimentConfig
from smd.runtime import runtime_from_env
from scripts.inference.launch_multi_agent_experiment import run_multi_agent_experiment


def _num_agents_for_instance(instance_name: str) -> int:
    lowered = instance_name.lower()
    if "two" in lowered:
        return 2
    if "three" in lowered:
        return 3
    if "six" in lowered:
        return 6
    if "nine" in lowered:
        return 9
    raise ValueError(f"Cannot infer num_agents from instance_name={instance_name!r}")


def _load_init_traj(*, map_name: str, sample_idx: int, num_agents: int):
    init_path = runtime_from_env().init4proj_root / f"{map_name}_init4proj_agent_{num_agents}.pkl"
    payload = pickle.load(open(init_path, "rb"))
    return payload[sample_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent inference with the DFM generator.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1)
    parser.add_argument("--map-name", type=str, default="instances_simple")
    parser.add_argument("--runtime-limit", type=int, default=100000)
    parser.add_argument(
        "--experiment-instance-names",
        nargs="+",
        default=["EnvEmptyNoWait2DRobotCompositeThreePlanarDiskRandom"],
    )
    parser.add_argument("--agents-max-speeds", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--rho-factor", type=float, default=1.05)
    parser.add_argument("--alm-iteration", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--projection-step", type=int, nargs="+", default=[15, 5])
    args = parser.parse_args()

    runtime = runtime_from_env()
    proj_params = {
        "agents_max_speeds": args.agents_max_speeds,
        "rho": args.rho,
        "rho_factor": args.rho_factor,
        "alm_iteration": args.alm_iteration,
        "tolerance": args.tolerance,
        "projection_step": args.projection_step,
    }

    for sample_idx in range(args.start_index, args.end_index):
        for instance_name in args.experiment_instance_names:
            num_agents = _num_agents_for_instance(instance_name)
            experiment_config = MultiAgentPlanningExperimentConfig()
            experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            experiment_config.instance_idx = sample_idx
            experiment_config.map_name = args.map_name
            experiment_config.res_base_dir = str(runtime.inference_root)
            experiment_config.num_agents_l = [num_agents]
            experiment_config.instance_name = instance_name
            experiment_config.init_traj4proj = _load_init_traj(
                map_name=args.map_name,
                sample_idx=sample_idx,
                num_agents=num_agents,
            )
            experiment_config.proj_params = proj_params
            experiment_config.stagger_start_time_dt = 0
            experiment_config.multi_agent_planner_class_l = ["SMDComposite"]
            experiment_config.single_agent_planner_class = "SMDEnsemble"
            experiment_config.runtime_limit = args.runtime_limit
            experiment_config.num_trials_per_combination = 1
            experiment_config.render_animation = True
            run_multi_agent_experiment(experiment_config)


if __name__ == "__main__":
    main()
