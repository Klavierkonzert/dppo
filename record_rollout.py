# Rollout of DPPO agent in the Kitchen env
import argparse
import os
import re
from collections import deque
from datetime import datetime

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import gym
import d4rl  # noqa: F401 - registers D4RL envs
import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

DEFAULT_VIDEO_DIR = "videos"
DEFAULT_MAX_EPISODE_STEPS = 280

from pathlib import Path


def register_resolvers():
    import math

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now().strftime(pattern), replace=True
    )

def resolve_checkpoint_path(path: str, _EXT: str='.pt') -> str:
    """Searches for `'.pt'` at the end of the file, and if NOT found - picks up the most recent `.pt` file. 
    
    :return: most recent model path, if the path points to a folder. Returns `path` without any changes otherwise."""
    ppath = Path(path).expanduser()
    if (ppath.is_file() and ppath.suffix==_EXT):
        return path
    else:
        # Check if path exists and is a directory
        if not ppath.is_dir():
            raise NotADirectoryError(f"{path} is neither a '.pt` file nor a valid directory")

        # Get subdirectories
        subdirs = [d for d in ppath.iterdir() if d.is_dir()]
        newest_subdir = ppath #default search dir
        if subdirs:
            # Pick most recently modified subdirectory
            newest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)

        #Search for files recursively
        files = [f for f in newest_subdir.rglob(f"*{_EXT}") if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No {_EXT} files found in {newest_subdir}")

        # by modification time (newest first)
        # Pick most recently modified file
        newest_file = str(max(files, key=lambda f: f.stat().st_mtime))
        print(f"[INFO]: resolved model path: {str(newest_file)}")
        return newest_file


def infer_env_name(path: str) ->str:
    match = re.search(r"(kitchen-(?:complete|partial|mixed)-v\d+)", path)
    if match:
        return match.group(1)
    match = re.search(r"([a-z0-9]+(?:-[a-z0-9]+)*-v\d+)", path)
    if match:
        return match.group(1)
    raise ValueError(f"Could not infer env name from path: {path}")


def infer_stage(path):
    path = path.lower()
    if "pretrain" in path:
        return "pretrain"
    if "finetune" in path:
        return "finetune"
    return "finetune"


def default_config_path(env_name, stage):
    if stage == "pretrain":
        return f"cfg/gym/pretrain/{env_name}/pre_diffusion_mlp.yaml"
    return f"cfg/gym/finetune/{env_name}/ft_ppo_diffusion_mlp.yaml"


def load_config(config_path, env_name, device, stage):
    if config_path is None:
        config_path = default_config_path(env_name, stage)
    cfg = OmegaConf.load(config_path)
    cfg.device = device
    if "env_name" in cfg:
        cfg.env_name = env_name
    if "env" in cfg and isinstance(cfg.env, str):
        cfg.env = env_name
    if "env" in cfg and not isinstance(cfg.env, str) and "name" in cfg.env:
        cfg.env.name = env_name
        cfg.env.n_envs = 1
        cfg.env.save_video = False
    if "normalization_path" not in cfg:
        data_dir = os.environ.get("DPPO_DATA_DIR")
        if data_dir is None:
            raise ValueError(
                "Config does not define normalization_path and DPPO_DATA_DIR is not set."
            )
        cfg.normalization_path = os.path.join(
            data_dir, "gym", env_name, "normalization.npz"
        )
    if "model" in cfg and "network_path" in cfg.model:
        cfg.model.network_path = None
    OmegaConf.resolve(cfg)
    return cfg


def default_output_path(out_dir, stage, env_name, ckpt_path):
    checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{env_name}_{run_dir}_{checkpoint_name}_{timestamp}.mp4"
    return os.path.join(out_dir, stage, filename)


def load_normalizer(path):
    data = np.load(path)
    return {
        "obs_min": data["obs_min"],
        "obs_max": data["obs_max"],
        "action_min": data["action_min"],
        "action_max": data["action_max"],
    }


def normalize_obs(obs, stats):
    return 2 * ((obs - stats["obs_min"]) / (stats["obs_max"] - stats["obs_min"] + 1e-6) - 0.5)


def unnormalize_action(action, stats):
    action = (action + 1) / 2
    return action * (stats["action_max"] - stats["action_min"]) + stats["action_min"]


def render_frame(env, width, height, camera_name):
    try:
        return format_frame(env.render(mode="rgb_array", width=width, height=height))
    except TypeError:
        pass
    except Exception:
        pass

    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None:
        try:
            frame = sim.render(width=width, height=height, camera_name=camera_name)
            return format_frame(frame[::-1])
        except Exception:
            frame = sim.render(width=width, height=height)
            return format_frame(frame[::-1])

    frame = env.render(mode="rgb_array")
    if frame is None:
        raise RuntimeError(
            "Kitchen did not return RGB frames. Try running with a display/X server, "
            "or set MUJOCO_GL=egl/osmesa in the environment before launching."
        )
    return format_frame(frame)


def format_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def build_model(cfg, ckpt_path, device, weights):
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location=device)
    if weights == "auto":
        weights = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[weights], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--config", type=str, default=None, help="Matching DPPO yaml config")
    parser.add_argument("--env", type=str, default=None, help="Override env name")
    parser.add_argument(
        "--stage",
        choices=["auto", "pretrain", "finetune"],
        default="auto",
        help="Controls default config and output subfolder.",
    )
    parser.add_argument(
        "--weights",
        choices=["auto", "model", "ema"],
        default="auto",
        help="Checkpoint weights to load. auto prefers EMA when present.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=str, default="fixed")
    args = parser.parse_args()

    register_resolvers()

    checkpoint_path = resolve_checkpoint_path(args.ckpt)
    env_name = args.env or infer_env_name(checkpoint_path)
    stage = infer_stage(checkpoint_path) if args.stage == "auto" else args.stage
    out_path = args.out or default_output_path(args.out_dir, stage, env_name, checkpoint_path)
    print(f"[INFO] Env: {env_name}")
    print(f"[INFO] Stage: {stage}")

    cfg = load_config(args.config, env_name, args.device, stage)
    stats = load_normalizer(cfg.normalization_path)
    model = build_model(cfg, checkpoint_path, args.device, args.weights)

    env = gym.make(env_name)
    env.seed(args.seed)
    obs = env.reset()

    obs_history = deque(maxlen=cfg.cond_steps)
    norm_obs = normalize_obs(obs, stats).astype(np.float32)
    for _ in range(cfg.cond_steps):
        obs_history.append(norm_obs)

    act_steps = cfg.get("act_steps", cfg.horizon_steps)
    max_episode_steps = (
        cfg.env.max_episode_steps
        if "env" in cfg and not isinstance(cfg.env, str) and "max_episode_steps" in cfg.env
        else DEFAULT_MAX_EPISODE_STEPS
    )
    max_steps = args.max_steps or max_episode_steps
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    total_reward = 0.0
    env_steps = 0
    done = False

    print(f"[INFO] Writing video: {out_path}")
    with imageio.get_writer(out_path, format="FFMPEG", fps=args.fps, codec=args.codec) as writer:
        writer.append_data(render_frame(env, args.width, args.height, args.camera))

        while not done and env_steps < max_steps:
            cond_np = np.stack(obs_history, axis=0)[None]
            cond = {"state": torch.from_numpy(cond_np).float().to(args.device)}

            with torch.no_grad():
                samples = model(cond=cond, deterministic=True, return_chain=False)
                action_chunk = samples.trajectories[0, :act_steps].detach().cpu().numpy()

            for action in action_chunk:
                raw_action = unnormalize_action(action, stats)
                obs, reward, done, _ = env.step(raw_action)
                total_reward += float(reward)
                env_steps += 1

                norm_obs = normalize_obs(obs, stats).astype(np.float32)
                obs_history.append(norm_obs)
                writer.append_data(render_frame(env, args.width, args.height, args.camera))

                if done or env_steps >= max_steps:
                    break

    env.close()
    print(f"[INFO] Done. steps={env_steps} total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
