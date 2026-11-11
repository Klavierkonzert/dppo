"""
Continue PPO fine-tuning from the latest checkpoint under ${DPPO_LOG_DIR}/gym-finetune.

This is a thin launcher around `script/run.py` that:
  1. Locates the most recent `state_<itr>.pt` (using the same logic as
     `record_rollout.resolve_checkpoint_path`).
  2. Invokes `script/run.py` with `base_policy_path=<that path>`, so the
     diffusion model's `__init__` loads the previous fine-tune weights instead
     of the original pretrain checkpoint.

Caveat: warm start only. Optimizer state, LR scheduler position, and the
training iteration counter restart from 0, and a new timestamped logdir is
created so the previous run is left intact.

Examples:
    # Default: continue the most recent kitchen finetune
    python continue_finetune.py

    # Different task / config
    python continue_finetune.py \\
        --config-name ft_ppo_diffusion_mlp \\
        --config-dir cfg/gym/finetune/kitchen-complete-v0

    # Forward extra Hydra overrides after `--`
    python continue_finetune.py -- wandb=null train.n_train_itr=500 seed=7
"""

import argparse
import os
import sys
from pathlib import Path

# Reuse the latest-checkpoint resolver already used by record_rollout.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_rollout import resolve_checkpoint_path


DEFAULT_CONFIG_NAME = "ft_ppo_diffusion_mlp"
DEFAULT_CONFIG_DIR = "cfg/gym/finetune/kitchen-complete-v0"


def default_search_root() -> str:
    log_dir = os.environ.get("DPPO_LOG_DIR")
    if not log_dir:
        raise EnvironmentError(
            "DPPO_LOG_DIR is not set. `source script/set_path.sh` or "
            "`export DPPO_LOG_DIR=/path/to/log` first."
        )
    return os.path.join(log_dir, "gym-finetune")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Explicit .pt path, or a directory to search. "
            "Defaults to ${DPPO_LOG_DIR}/gym-finetune."
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

    search_root = args.ckpt or default_search_root()
    ckpt_path = resolve_checkpoint_path(search_root)
    print(f"[INFO] Continuing from checkpoint: {ckpt_path}")

    extras = list(args.overrides)
    if extras and extras[0] == "--":
        extras = extras[1:]

    cmd = [
        sys.executable,
        "script/run.py",
        f"--config-name={args.config_name}",
        f"--config-dir={args.config_dir}",
        f"base_policy_path={ckpt_path}",
        *extras,
    ]
    print("[INFO] Launch command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return

    # Hydra's @hydra.main reads config_path from os.getcwd() in script/run.py,
    # so the user must invoke this from the repo root (same as script/run.py).
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
