"""
Train a Franka-Kitchen policy *from scratch* on a chosen set of tasks.

This is the rung below `run_pretrained.py`: there is **no pretrained model and
no offline demo data**. The policy starts from random initialization and the
only thing shaping its behavior is the task-completion reward, which you
restrict with `--tasks`. That is what gives you full control over which
appliances the robot learns to manipulate -- a behavior-cloning prior (used by
`run_pretrained.py`) keeps producing the demonstrated motions regardless of the
reward, whereas from scratch the reward is the sole driver.

Default method is pure online SAC (`sac_mlp`, no offline buffer). Passing
`--offline-data` switches to RLPD (`rlpd_mlp`), which *also* trains on the
kitchen demo dataset (mixed 50/50 into every update) while still using **no
pretrained policy**. RLPD only kicks in when the offline dataset
(`$DPPO_DATA_DIR/gym/<env_name>/train.npz`) actually exists; otherwise the run
falls back to pure SAC with a warning.

Note: RLPD's offline samples carry the dataset's *original* task rewards (the 4
demonstrated tasks for kitchen-complete: microwave, kettle, light switch, slide
cabinet), not your `--tasks` restriction. So offline data helps most when
`--tasks` overlaps the demonstrated tasks; for a disjoint set it can reintroduce
the demonstrated behaviors and work against "full control".

Caveat: kitchen rewards are sparse (+1 per completed appliance). Pure
from-scratch RL explores poorly under sparse reward, so a run may need a long
time -- or dense reward shaping -- before it solves anything. CPU is the
bottleneck on this machine, so keep `env.n_envs` small.

Valid task elements:
    bottom burner, top burner, light switch, slide cabinet,
    hinge cabinet, microwave, kettle

Examples:
    # SAC from scratch, only ever rewarded for the two burners
    python run_scratch.py --tasks "bottom burner,top burner"

    # single-task control
    python run_scratch.py --tasks microwave

    # also learn from offline demo data (RLPD), still no pretrained model
    python run_scratch.py --tasks microwave,kettle --offline-data

    # end the episode as soon as the requested tasks are done
    python run_scratch.py --tasks microwave,kettle --terminate-on-complete

    # forward extra Hydra overrides after `--`
    python run_scratch.py --tasks microwave -- wandb=null seed=7 env.n_envs=4
"""

import argparse
import json
import os
import sys

DEFAULT_CONFIG_NAME = "sac_mlp"
OFFLINE_CONFIG_NAME = "rlpd_mlp"
DEFAULT_CONFIG_DIR = "cfg/gym/scratch/kitchen-complete-v0"


def offline_dataset_path(config_dir: str):
    """Expected offline dataset path for a scratch config dir, or None if
    DPPO_DATA_DIR is unset. env_name is the config dir's basename
    (e.g. cfg/gym/scratch/kitchen-complete-v0 -> kitchen-complete-v0)."""
    data_dir = os.environ.get("DPPO_DATA_DIR")
    if not data_dir:
        return None
    env_name = os.path.basename(os.path.normpath(config_dir))
    return os.path.join(data_dir, "gym", env_name, "train.npz")

# Canonical kitchen elements (mirrors d4rl OBS_ELEMENT_INDICES). Used for
# alias/preset handling and friendly errors; script/run.py re-validates.
KITCHEN_ELEMENTS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

TASK_PRESETS = {
    "complete": ["microwave", "kettle", "light switch", "slide cabinet"],
    "default": ["microwave", "kettle", "light switch", "slide cabinet"],
    "complete6": [
        "microwave",
        "kettle",
        "light switch",
        "slide cabinet",
        "bottom burner",
        "top burner",
    ],
    "all": list(KITCHEN_ELEMENTS),
}


def parse_tasks(spec: str) -> list:
    """Turn a --tasks value into a validated list of canonical element names."""
    key = spec.strip().lower()
    if key in TASK_PRESETS:
        return list(TASK_PRESETS[key])

    tasks = []
    for raw in spec.split(","):
        name = raw.strip().lower().replace("_", " ")
        if not name:
            continue
        if name not in KITCHEN_ELEMENTS:
            raise SystemExit(
                f"[ERROR] Unknown kitchen task '{raw.strip()}'.\n"
                f"        Valid elements: {', '.join(KITCHEN_ELEMENTS)}\n"
                f"        Valid presets:  {', '.join(sorted(TASK_PRESETS))}"
            )
        tasks.append(name)

    if not tasks:
        raise SystemExit("[ERROR] --tasks resolved to an empty set.")
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help=(
            "Comma-separated kitchen elements (spaces or underscores), or a "
            "preset: complete (4), complete6 (6), all (7). "
            f"Elements: {', '.join(KITCHEN_ELEMENTS)}."
        ),
    )
    parser.add_argument(
        "--offline-data",
        action="store_true",
        help="Also train on offline demo data via RLPD (still no pretrained "
        "model). Used only if the dataset exists; otherwise falls back to SAC.",
    )
    parser.add_argument(
        "--terminate-on-complete",
        action="store_true",
        help="End the episode once every requested task is done "
        "(default: keep running until max_episode_steps).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help=f"Hydra config name. Default: {DEFAULT_CONFIG_NAME}, or "
        f"{OFFLINE_CONFIG_NAME} when --offline-data is set. An explicit value "
        "overrides --offline-data.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=DEFAULT_CONFIG_DIR,
        help=f"Hydra config dir (default: {DEFAULT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command and exit without running.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides forwarded to script/run.py. "
        "Prefix with `--` to separate them, e.g. `-- wandb=null seed=7`.",
    )
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    print(f"[INFO] Training FROM SCRATCH (no pretrained model).")
    print(f"[INFO] Kitchen tasks ({len(tasks)}): {tasks}")

    # Resolve which algorithm/config to run. An explicit --config-name wins;
    # otherwise --offline-data picks RLPD (only if the dataset is present).
    ds_path = offline_dataset_path(args.config_dir)
    ds_available = ds_path is not None and os.path.exists(ds_path)

    if args.config_name:
        config_name = args.config_name
        wants_offline = config_name == OFFLINE_CONFIG_NAME
        if wants_offline and not ds_available:
            raise SystemExit(
                f"[ERROR] {config_name} needs offline data, but none was found at\n"
                f"        {ds_path or '<DPPO_DATA_DIR unset>'}\n"
                f"        Provide the dataset there, or omit --config-name to run "
                f"pure SAC from scratch."
            )
    elif args.offline_data:
        if ds_available:
            config_name = OFFLINE_CONFIG_NAME
        else:
            print(
                "[WARN] --offline-data requested but no dataset found at\n"
                f"       {ds_path or '<DPPO_DATA_DIR unset>'}\n"
                "       Falling back to pure SAC from scratch (no offline data)."
            )
            config_name = DEFAULT_CONFIG_NAME
    else:
        config_name = DEFAULT_CONFIG_NAME

    if config_name == OFFLINE_CONFIG_NAME:
        print(f"[INFO] Offline data: ON (RLPD) <- {ds_path}")
        print(
            "[INFO] Note: offline rewards reflect the dataset's original tasks, "
            "not --tasks; most useful when --tasks overlaps the demonstrated tasks."
        )
    else:
        print("[INFO] Offline data: OFF (pure online SAC)")

    extras = list(args.overrides)
    if extras and extras[0] == "--":
        extras = extras[1:]

    cmd = [
        sys.executable,
        "script/run.py",
        f"--config-name={config_name}",
        f"--config-dir={args.config_dir}",
        *extras,
    ]

    # Forward task config via env vars to dodge Hydra quoting of spaces.
    env = dict(os.environ)
    env["DPPO_KITCHEN_TASKS"] = json.dumps(tasks)
    env["DPPO_KITCHEN_TERMINATE_ON_COMPLETE"] = "1" if args.terminate_on_complete else "0"

    print("[INFO] Launch command:")
    print("  DPPO_KITCHEN_TASKS='" + env["DPPO_KITCHEN_TASKS"] + "' \\")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return

    # Hydra's @hydra.main reads config_path from os.getcwd() in script/run.py,
    # so invoke this from the repo root (same as script/run.py).
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
