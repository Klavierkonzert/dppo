from pathlib import Path
import subprocess

dir_path = 'log/gym-finetune/kitchen-complete-v0_ppo_diffusion_mlp_ta4_td20_tdf10/2026-05-21_14-53-48_42/checkpoint/state_29.pt'
dir_path = 'log/gym-finetune/kitchen-complete-v0_ppo_diffusion_mlp_ta4_td20_tdf10'
dir_path = Path('log/gym-finetune/kitchen-mixed-v0_ppo_diffusion_mlp_ta4_td20_tdf10')
latest_dir = max(dir_path.iterdir()) / 'checkpoint'
latest_state = max(latest_dir.iterdir())

command = ['python', 'record_rollout.py']
command.extend(['--ckpt', str(latest_state.absolute())])
command.extend(['--out', 'kitchen_latest.mp4'])
command.extend(['--device', 'cuda:0'])

print(f'Running on {str(latest_state.absolute())}')

subprocess.run(command)
