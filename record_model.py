from pathlib import Path
import subprocess
import sys

latest_state = Path(sys.argv[1])

command = ['python', 'record_rollout.py']
command.extend(['--ckpt', str(latest_state.absolute())])
command.extend(['--out', 'kitchen_latest.mp4'])
command.extend(['--device', 'cuda:0'])

print(f'Running on {str(latest_state.absolute())}')

subprocess.run(command)
