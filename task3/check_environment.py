"""Run from the repo root: python task3/check_environment.py."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

repo = Path(__file__).resolve().parents[1]
notebook = json.loads((repo / 'task3/reinforcement_learning.ipynb').read_text())
exec(''.join(notebook['cells'][5]['source']))
exec(''.join(notebook['cells'][7]['source']))
config.action.n_frames = 2
env = PickAndPlace(str(repo / 'assets/descriptions/DropCubeInBinEnv.xml'), config)
reset = jax.jit(env.reset)
a, b = (reset(jax.random.PRNGKey(seed)) for seed in (0, 1))
repeat = reset(jax.random.PRNGKey(0))
assert np.allclose(a.pipeline_state.q, repeat.pipeline_state.q)
assert not np.allclose(a.pipeline_state.q[:7], b.pipeline_state.q[:7])
for body in (env.cube_body_id, env.bin_body_id):
    assert not np.allclose(a.pipeline_state.x.pos[body], b.pipeline_state.x.pos[body])
next_state = jax.jit(env.step)(a, jax.numpy.zeros(env.action_size))
assert np.isfinite(np.asarray(next_state.obs['state'])).all()
assert np.allclose(a.pipeline_state.mocap_pos, next_state.pipeline_state.mocap_pos)

# Controlled contact and pose checks reject hover, held, moving and outside cubes.
p = a.pipeline_state
links = (jax.numpy.array([env.left_finger_id, env.right_finger_id]),
         jax.numpy.array([env.cube_body_id, env.cube_body_id]))
free = SimpleNamespace(link_idx=links, dist=jax.numpy.ones(2))
held = SimpleNamespace(link_idx=links, dist=jax.numpy.zeros(2))
bin_pos = p.x.pos[env.bin_body_id]
def pose(height, contact=free, speed=0.0, dx=0.0, finger_width=0.04):
    pos = p.x.pos.at[env.cube_body_id].set(bin_pos + jax.numpy.array([dx, 0, height]))
    return SimpleNamespace(x=p.x.replace(pos=pos), contact=contact,
                           q=p.q.at[7:9].set(finger_width),
                           qd=jax.numpy.zeros(env.nv).at[-6].set(speed))
assert bool(env.check_success(pose(0.030)))
assert not bool(env.check_success(pose(0.060)))
assert not bool(env.check_success(pose(0.030, held)))
assert not bool(env.check_success(pose(0.030, SimpleNamespace(link_idx=links, dist=jax.numpy.array([0.0, 0.1])))))
assert not bool(env.check_success(pose(0.030, finger_width=0.02)))
assert not bool(env.check_success(pose(0.030, speed=0.10)))
assert not bool(env.check_success(pose(0.030, dx=0.03)))
# Placement bonus requires a preceding grasped lift, not a lucky collision.
original_step = env.pipeline_step
env.pipeline_step = lambda previous, ctrl: pose(0.030)
assert not bool(env.step(a, jax.numpy.zeros(8)).info['success'])
env.pipeline_step = lambda previous, ctrl: pose(0.075, held, finger_width=0.02)
lifted = env.step(a, jax.numpy.zeros(8))
assert bool(lifted.info['has_lifted']) and not bool(lifted.info['success'])
env.pipeline_step = lambda previous, ctrl: pose(0.030)
assert bool(env.step(lifted, jax.numpy.zeros(8)).info['success'])
env.pipeline_step = original_step
# Reward priorities: hover stays small; lift requires grasp; placement dominates.
info = {**a.info, 'cube_init_z': p.x.pos[env.cube_body_id, 2] - 0.15}
action = jax.numpy.zeros(env.action_size)
hover, _ = env._compute_reward(p, info, action)
pickup, _ = env._compute_reward(p, {**info, 'is_grasped': jax.numpy.array(True)}, action)
placement, _ = env._compute_reward(p, {**info, 'success': jax.numpy.array(True)}, action)
assert 0 < float(hover) <= 0.1
assert 5.5 <= float(pickup - hover) <= 7.5
assert np.isclose(float(placement - hover), 5000.0)
# Exercise the real Brax autoreset path: custom counters/history must reset too.
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
env.max_steps = 2
wrapped = AutoResetWrapper(EpisodeWrapper(env, episode_length=2, action_repeat=1))
w = jax.jit(wrapped.reset)(jax.random.PRNGKey(0))
w = w.replace(info={**w.info, 'has_lifted': jax.numpy.array(True)})
wrapped_step = jax.jit(wrapped.step)
home_ctrl = jax.numpy.concatenate([env.init_q[:7], jax.numpy.array([255.0])])
home_action = 2 * (home_ctrl - env.ctrl_range[:, 0]) / (env.ctrl_range[:, 1] - env.ctrl_range[:, 0]) - 1
w = wrapped_step(wrapped_step(w, home_action), home_action)
assert bool(w.done)
w = wrapped_step(w, home_action)
assert not bool(w.done) and int(w.info['step']) == 1
assert not bool(w.info['has_lifted'])
assert config.ppo_agent.num_resets_per_eval == 0
print('PASS: seeded resets, stepping, bin persistence, grasp-lift-release sequence, reward priorities and autoreset')
