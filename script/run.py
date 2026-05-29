"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import json
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)
from d4rl.kitchen.kitchen_envs import (
    OBS_ELEMENT_INDICES,
    KitchenMicrowaveKettleLightSliderV0,
    KitchenMicrowaveKettleBottomBurnerLightV0,
)

# Default Franka-Kitchen task set when none is requested. Matches the original
# 6-element experiment so existing launch commands behave identically.
DEFAULT_KITCHEN_TASKS = [
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
    "bottom burner",
    "top burner",
]


def configure_kitchen_tasks(cfg):
    """Choose which kitchen appliances count toward reward/success.

    Resolution order: ``DPPO_KITCHEN_TASKS`` env var (JSON list, set by
    ``run_pretrained.py``) > ``cfg.kitchen_task_elements`` > the 6-element
    default. The success threshold is kept in sync (one point per appliance).
    No-op for non-kitchen envs.
    """
    env_name = cfg.get("env_name") or cfg.get("env")
    if env_name is None or "kitchen" not in str(env_name):
        return

    raw = os.environ.get("DPPO_KITCHEN_TASKS")
    if raw:
        tasks = json.loads(raw)
        explicit = True
    elif cfg.get("kitchen_task_elements"):
        tasks = list(cfg.kitchen_task_elements)
        explicit = True
    else:
        tasks = list(DEFAULT_KITCHEN_TASKS)
        explicit = False

    unknown = [t for t in tasks if t not in OBS_ELEMENT_INDICES]
    if unknown:
        raise ValueError(
            f"Unknown kitchen task element(s) {unknown}. "
            f"Valid elements: {sorted(OBS_ELEMENT_INDICES)}"
        )

    terminate = os.environ.get("DPPO_KITCHEN_TERMINATE_ON_COMPLETE", "0") == "1"
    for kitchen_cls in (
        KitchenMicrowaveKettleLightSliderV0,
        KitchenMicrowaveKettleBottomBurnerLightV0,
    ):
        kitchen_cls.TASK_ELEMENTS = list(tasks)
        kitchen_cls.TERMINATE_ON_TASK_COMPLETE = terminate

    # Each completed appliance yields +1 reward, so completing all == len(tasks).
    if explicit and "env" in cfg and "best_reward_threshold_for_success" in cfg.env:
        cfg.env.best_reward_threshold_for_success = len(tasks)

    threshold = (
        cfg.env.best_reward_threshold_for_success
        if "env" in cfg and "best_reward_threshold_for_success" in cfg.env
        else "n/a"
    )
    log.info(
        f"Kitchen tasks: {tasks} "
        f"(terminate_on_complete={terminate}, success_threshold={threshold})"
    )

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # set kitchen task elements before the (forked) env workers are created
    configure_kitchen_tasks(cfg)

    print("DEBUG: CFG OBS DIM:", cfg.obs_dim)

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = os.path.dirname(cfg.train_dataset_path)
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        gdown.download_folder(url=download_url, output=download_target)

    # For for-tuning: download normalization if needed
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(
            f"Downloading normalization statistics from {download_url} to {download_target}"
        )
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # For for-tuning: download checkpoint if needed
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(
                f"Unknown checkpoint path. Did you specify the correct path to the policy you trained?"
            )
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)

    #  load checkpoint
    # only main process should load
    if hasattr(agent, "model") and "checkpoint" in cfg:
        import torch

        ckpt_path = os.path.abspath(cfg.checkpoint)
        print(f"Loading checkpoint from {ckpt_path}")

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu"   # IMPORTANT
        )

        agent.model.load_state_dict(ckpt["model"])

        # THEN move model to GPU
        agent.model.to(cfg.device)
    agent.run()


if __name__ == "__main__":
    main()
