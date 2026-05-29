"""
Fine-tune the *pre-trained* diffusion policy on a chosen set of Franka-Kitchen
tasks.

Unlike `continue_finetune.py` (which warm-starts from the latest *fine-tuned*
checkpoint), this launcher leaves `base_policy_path` at the config default,
which points at the pre-training checkpoint
(`${DPPO_LOG_DIR}/gym-pretrain/.../state_8000.pt`). It is a thin wrapper around
`script/run.py` that selects which kitchen appliances count toward reward and
success via `--tasks`.

The task set is forwarded to `script/run.py` through the `DPPO_KITCHEN_TASKS`
environment variable (JSON), which avoids Hydra CLI quoting problems with the
spaces in element names ("light switch", "slide cabinet", ...).

Valid elements:
    bottom burner, top burner, light switch, slide cabinet,
    hinge cabinet, microwave, kettle

Examples:
    # 4 tasks the kitchen-complete dataset actually demonstrates
    python run_pretrained.py --tasks complete

    # original 6-task experiment (the previous hardcoded default)
    python run_pretrained.py --tasks complete6

    # a custom subset (comma-separated; spaces or underscores both work)
    python run_pretrained.py --tasks "microwave,kettle,hinge cabinet"
    python run_pretrained.py --tasks microwave,kettle,hinge_cabinet

    # all seven appliances
    python run_pretrained.py --tasks all

    # forward extra Hydra overrides after `--`
    python run_pretrained.py --tasks microwave,kettle -- wandb=null seed=7 train.n_train_itr=300
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse the latest-checkpoint resolver already used by record_rollout.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_rollout import resolve_checkpoint_path


DEFAULT_CONFIG_NAME = "ft_ppo_diffusion_mlp"
DEFAULT_CONFIG_DIR = "cfg/gym/finetune/kitchen-complete-v0"

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
    # what the kitchen-complete dataset actually demonstrates
    "complete": ["microwave", "kettle", "light switch", "slide cabinet"],
    "default": ["microwave", "kettle", "light switch", "slide cabinet"],
    # the previous hardcoded 6-task experiment in script/run.py
    "complete6": [
        "microwave",
        "kettle",
        "light switch",
        "slide cabinet",
        "bottom burner",
        "top burner",
    ],
    # everything the simulator supports
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
        default="complete6",
        help=(
            "Comma-separated kitchen elements (spaces or underscores), or a "
            "preset: complete (4 dataset tasks), complete6 (default, the "
            "original 6-task set), all (7). "
            f"Elements: {', '.join(KITCHEN_ELEMENTS)}."
        ),
    )
    parser.add_argument(
        "--terminate-on-complete",
        action="store_true",
        help="End the episode once every requested task is done "
        "(default: keep running until max_episode_steps).",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=None,
        help=(
            "Explicit pre-train .pt path, or a directory to search for the "
            "newest one. Defaults to the config's base_policy_path "
            "(the pre-training checkpoint)."
        ),
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help=f"Hydra config name (default: {DEFAULT_CONFIG_NAME}).",
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
    print(f"[INFO] Kitchen tasks ({len(tasks)}): {tasks}")

    extras = list(args.overrides)
    if extras and extras[0] == "--":
        extras = extras[1:]

    cmd = [
        sys.executable,
        "script/run.py",
        f"--config-name={args.config_name}",
        f"--config-dir={args.config_dir}",
        *extras,
    ]

    if args.base_ckpt:
        ckpt_path = resolve_checkpoint_path(args.base_ckpt)
        print(f"[INFO] Base (pre-train) checkpoint: {ckpt_path}")
        cmd.append(f"base_policy_path={ckpt_path}")
    else:
        print("[INFO] Base checkpoint: config default (pre-training checkpoint)")

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
