"""From repo root: python task3/resume_training.py CHECKPOINT [--check]."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
checkpoint = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
os.chdir(repo)
notebook = json.loads((repo / 'task3/reinforcement_learning.ipynb').read_text())
# ponytail: reuse notebook cells; update these indices if cells move.
for i in (5, 7, 9):
    exec(''.join(notebook['cells'][i]['source']))

from brax.io import model
from brax.training.acme import running_statistics
import numpy as np

resume_params = model.load_params(str(checkpoint))
# Bin coordinates were constant during the previous training stage. Prevent
# centimeter reset jitter from becoming normalized inputs of magnitude 10,000.
std = jax.tree.map(lambda s: s.at[37:40].set(jp.maximum(s[37:40], 0.01)), resume_params[0].std)
normalizer = resume_params[0].replace(
    count=jp.array(1000.0), std=std,
    summed_variance=jax.tree.map(lambda s: 1000.0 * s**2, std))
resume_params = (normalizer, resume_params[1])
config.ppo_agent.num_timesteps = 3_000_000
config.ppo_agent.num_evals = 9  # 8 batches: 3,276,800 additional transitions.
env = PickAndPlace(XML_PATH, config)
eval_env = PickAndPlace(XML_PATH, config)
sample = jax.jit(env.reset)(jax.random.PRNGKey(config.seed))
normalized = running_statistics.normalize(sample.obs, normalizer)
net = make_networks_factory(env.observation_size, env.action_size,
                           preprocess_observations_fn=running_statistics.normalize)
action, _ = ppo_networks.make_inference_fn(net)(resume_params)(sample.obs, jax.random.PRNGKey(0))
assert action.shape == (8,) and np.isfinite(np.asarray(action)).all()
assert all(np.isfinite(np.asarray(v)).all() for v in normalized.values())
assert float(jp.max(jp.abs(normalized['state'][37:40]))) < 2
print('RESUME CHECK PASSED: randomized reset, finite normalized inputs/actions, bin std >= .01', flush=True)
if '--check' in sys.argv[2:]:
    raise SystemExit(0)
if not any(device.platform == 'gpu' for device in jax.devices()):
    raise SystemExit('GPU required for continuation; checkpoint remains saved.')

training = ''.join(notebook['cells'][13]['source']).replace('task3_corrected', 'task3_resumed')
training = training.replace(
    'Starting corrected-controls PPO: 10,000,000 requested transitions, 26 evals, 10,240,000 batch-rounded transitions',
    'Resuming PPO: randomized reset, strict release, 1000 controls, 3,276,800 additional transitions')
training = training.replace('ppo.train(environment=env,',
                            'ppo.train(restore_params=resume_params, restore_value_fn=False, environment=env,')
exec(training)
