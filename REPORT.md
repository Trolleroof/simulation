# Simulation onboarding: results and experience

**Date:** October 1, 2026
**Repository:** `Trolleroof/simulation`, branch `ml_onboarding`
**Status:** Tasks 1 and 2 have measured results. Task 3 has a rare learned placement and a matched random-policy comparison; the requested continued training and randomized-reset validation remain incomplete.

## 1. Scene construction

I composed the Panda arm, table, free cube, and bin in `DropCubeInBinEnv.xml`, added a fixed camera, and defined the required `home` keyframe. The arm begins at the assignment's joint values, with both fingers open at 0.04 m. The scene compiles in MuJoCo and is used by both subsequent tasks.

The original fingertip pad collision classes used `contype=0` and `conaffinity=0`, preventing contact with the cube. I enabled contact and set pad friction to 5. The bin is at `(0.18, 0.0, 0.765)`, within reach of the arm. This shorter transfer improved the planner's ability to keep the cube in its grip.

## 2. Imitation learning

### Implementation and method

The Gymnasium wrapper exposes 31 observation values (positions and velocities) and 8 actuator commands. The MPlib expert approaches the cube, lowers the open gripper, closes it, lifts, transfers over the bin, and releases. Stored expert actions are normalized to `[-1, 1]`; policy outputs are converted back to actuator ranges, including the gripper's `0..255` command.

I collected demonstrations in Linux with MuJoCo and MPlib. The current placement condition requires the cube center to be within 0.020 m of the bin center in XY, between 0.026 and 0.036 m above the bin origin, released from the fingers, and moving slower than 0.05 m/s. This rejects a cube simply held above the bin.

### Results

| Measure | Result |
|---|---:|
| Expert attempts, reset seeds 0–49 | 50 |
| Successful demonstrations | 42 (84%) |
| Observation/action pairs | 11,006 |
| Dataset dimensions | observations 31, actions 8 |
| Collection time | 13.52 seconds |
| BC architecture | two 256-unit Tanh layers, Gaussian mean and learned log standard deviation |
| Training | 50 epochs, Adam, learning rate 0.0003, batch size 256 |
| Final training time | 14.66 seconds |
| NLL at epochs 5 / 50 | -0.5922 / -5.1997 |
| Selected epoch-50 policy evaluation | 0/5 successful placements, seeds 1000–1004 |

The dataset was saved and loaded through `TrajectoryDataset`; the actor trained, saved, reloaded, and ran in the simulator. All ten periodic two-episode evaluations in the first run also had zero success. The first run's tie handling retained its epoch-5 checkpoint when every success score was zero. I changed checkpoint selection to prefer lower imitation loss among equal success scores, reran training, and verified that the selected epoch-50 policy reloaded and completed five evaluation episodes. It also achieved zero placements. Evaluation now allows 500 steps; the former 100-step limit was shorter than the expert's approximately 250-step trajectory.

### Experience and limitation

Enabling fingertip contacts addressed the initial physical failure. Before the stricter release check, the tuned expert completed 5/5 seeds at planner speed 0.8, versus 4/5 at speed 0.6. Friction 10 failed in the single tested seed, so I retained friction 5. These small checks do not establish broad reliability; the later 42/50 demonstration collection provides stronger evidence.

BC training loss decreased while rollout success remained zero. This is the observed limitation: fitting expert actions does not ensure that the policy completes a closed-loop sequence. Compounding action errors and leaving the demonstrated state distribution are plausible explanations, but this run does not isolate their causal contribution.

## 3. Reinforcement learning

### Implementation and debugging

I implemented a Brax `PipelineEnv` backed by MJX, with state and privileged-state observations, grasp and placement predicates, reward components, PPO checkpointing, and rollout rendering.

Several errors affected the early policy:

- MuJoCo body IDs include the world body; Brax body transforms omit it. I corrected the index offset.
- The reward used the palm origin instead of the fingertip center. It now uses `panda_hand_tcp`, 0.1034 m below the palm origin in the home pose.
- PPO's normalized actions were passed directly to differently scaled actuators. The environment now converts each action to its actuator's range.
- The callback read a missing `eval/success` key. It now reads `eval/episode_success`.
- Excessive evaluation frequency rounded a requested 10M-step run up to approximately 40.55M. With 26 evaluations, the full configuration executes 25 batches of 409,600 transitions, or 10.24M total.

### Current reward

| Component | Weight | Meaning |
|---|---:|---|
| Reach | 0.1 | Small dense feedback for approaching the cube |
| Grasp | 0.5 | Both fingers contacting the cube within 1 mm |
| Lift | up to 5 | Height gained while grasped |
| Transport | up to 2 | Moving toward the bin while grasped |
| Placement | 5,000 | Terminal bonus after a grasped lift and released placement |
| Action cost | 0.0001 × mean squared action | Small regularizer |

The interrupted Colab stage used placement 500, discount 0.97, and reward scaling 0.1. At that discount, a bonus delayed by 344 controls contributes only about 0.014 points before scaling. The continuation uses placement 5,000, discount 0.995, and scaling 0.01. The 5,000-point terminal bonus exceeds the infinite discounted continuing-reward bound of 1,520; scaling keeps the immediate terminal value at 50. These changes make delayed completion more visible to PPO but do not guarantee learning.

### Measured results

The earlier full run completed 10.24M transitions in 786 seconds on a Colab T4, but its rollout failed. Its success callback was incorrect, so its printed evaluation success values are not reliable. A later corrected-controls diagnostic also failed to lift or place the cube.

Colab disconnected the contact-based reward run at its runtime limit after the last visible evaluation at 7,372,800 transitions. The best checkpoint at 6,963,200 transitions is preserved locally for continuation. At 6,963,200 transitions, average evaluation episode reward was 62.084 and the earlier, broader success predicate reported 1/128 episodes (0.78125%). Its first nonzero evaluation was at 5,324,800 transitions. That broad metric alone does not establish released placement; the independent stricter evaluation is reported below. At 2,048,000 transitions, average bilateral contact duration increased from 0.820 to 7.047 steps per 500-step episode, and accumulated lift signal increased from 0.0052 to 0.1902. These are accumulated episode metrics, not pickup success rates.

An independent deterministic rollout of the 2,048,000-transition checkpoint, with the same fixed reset as training and the checkpoint's observation normalization, approached within 0.101 m of the cube. It had no bilateral grasp or lift, and failed placement. The video contains 101 frames at 10 fps (10.1 seconds).

I initially omitted observation normalization in a local replay. That produced a misleading close-contact result. The corrected video and result file replace that replay; the earlier 0.014 m/contact claim is withdrawn. Colab's training measurements used the correct preprocessing.

### Reset randomization and stronger validation

The saved notebook now varies arm joints by up to 0.02 radians and cube/bin XY by up to 0.01 m per seeded reset. The bin is a mocap body, preserving its randomized pose through MJX stepping without changing policy observation dimensions. The saved success predicate requires the slow cube to be near the bin bottom, neither finger contacting it, and both fingers open beyond 0.03 m. The step function also requires a preceding bilateral grasp with cube rise greater than 0.025 m in the same episode. This rejects holding it above the rim, against one finger, or knocking it into the bin without a meaningful grasped lift.

`python task3/check_environment.py` passed checks for seeded reproducibility, distinct arm/cube/bin positions, finite stepping, bin pose persistence, acceptance of released placement, and rejection of bilateral contact, one-finger contact, closed fingers, elevated, moving, and outside cubes.

The interrupted run used fixed reset positions and the earlier broader success predicate. These changes are intended for the continuation. A normalized-policy comparison with randomized resets produced 0/5 placements both for random initialization and the 2,048,000-step policy. Randomizing formerly constant bin features also creates a normalization distribution shift, so the policy needs training with the randomized environment before claiming generalization.

### Expert trajectory physics check

To check physical feasibility without using demonstrations for PPO training, I replayed one successful native MuJoCo expert trajectory through MJX, matching its 20 ms control targets with two MJX controls per target. I then held the final open-gripper command to allow release and settling. With the current 2 ms physics step and 10 ms control interval, MJX lifted the cube by 0.126 m, recorded 273 bilateral-contact controls, and passed the stricter released-placement check after 574 total controls (5.74 seconds). With a 1 ms physics step at the same control rate, it also passed after 547 controls. These are two checks of one reference trajectory, not a reliability estimate.

This shows that pickup and placement are physically possible under MJX. It also shows that the current 500-control episode cannot complete this particular reference sequence. The saved notebook now uses 1,000 controls per episode and its rollout loops use the same configured limit. Local environment checks passed again after the change. The live PPO run still uses 500 controls; this episode-budget revision must be used in the next training stage. The reference actions are only a physics diagnostic and were not used to train PPO weights.

### Rejected shortcuts

With the normalized 4,096,000-transition policy and fixed training resets, extending evaluation to 1,000 controls still yielded 0/5 placements. Requested arm targets changed by about 1.5 radians per 10 ms control, while the expert targets changed by about 0.0016 radians per 20 ms control. These are requested targets, not actual joint velocities. An experimental target speed limit of 0.8 radians/s, applied only during evaluation, yielded 0/3 grasps or placements; it was not adopted.

Reducing MJX contact slots from 668 to 128 looked safe in fixed-pose comparisons: no active contact pair was missing at the sampled poses. However, a paired expert trajectory passed with the original slots and failed with the cap, with cube positions differing by up to 0.148 m. I rejected the cap and retained the original contact configuration.

### Independent checkpoint evaluation

The strict evaluation of the 5,324,800-transition checkpoint found 1/128 released placements over 1,000 controls with fixed training resets. Seed 84 put the cube in the bin, with open fingers and no contact, at control 344. Its cube rose 0.073 m overall but did not reach 0.05 m while simultaneously reporting bilateral grasp. Native forward contact reconstruction of the recorded poses showed grasped rise up to 0.039 m. The recorded motion briefly lifts and drops the cube into the bin; it is rough and unreliable. Five other episodes achieved grasped rise above 0.05 m without placement.

A matched 128-episode comparison using a randomly initialized Gaussian policy, reset seeds 0–127, the same fixed scene, 1,000-control limit, and strict release check produced 0/128 placements. The trained checkpoint's observed placement rate is 0.78125%, versus 0% for that random policy. One successful event does not establish a reliable or statistically robust improvement.

I subsequently replayed seeds 80–95 with the new lift-before-release gate. Seed 84 still passed at control 344. Its q trajectory is bit-for-bit identical to the previous trace, and MJX's bilateral contact trace confirms grasped cube rise of 0.0394 m before release. In a separate 32-episode randomized-reset check of the 6,963,200 checkpoint with the bin normalization floor, one cube reached the bin but rose only 0.0072 m while grasped. That event does not meet the new pickup requirement and is not reported as a successful pick-and-place.

### Episode lifecycle correction

With 1,024 environments and 409,600 transitions per evaluation batch, each training batch advances an environment only 400 controls. The prior `num_resets_per_eval=1` reset every environment at that boundary. The continuation sets it to zero so episodes can span batches and reach the 1,000-control budget. Brax's autoreset wrapper restores the physical state but preserves custom information fields. I therefore clear the custom step counter and lift latch using the wrapper's previous-episode flag. A runnable check exercises the real wrapper through timeout and verifies that the next episode starts at step one with no remembered lift.

### Runtime interruption and continuation

Colab refused a new GPU connection because the account had reached its GPU usage limit. The fork contains the 6,963,200-transition checkpoint and `task3/resume_training.py`. Its local `--check` preflight passed: the checkpoint loads, randomized reset observations and policy actions are finite, and randomized bin inputs stay within the checked normalized range. The continuation uses the strict release predicate, 1,000-control episodes, and 3,276,800 additional transitions. It restores the actor and recalibrates the previously constant bin features with standard deviation at least 0.01 m and a 1,000-sample statistics prior; the value function and optimizer start fresh. No demonstrations are used to train PPO.

From the fork root, run `python task3/resume_training.py results/task3_policy_pickplace_6963200` for a GPU. A corresponding guarded continuation cell is saved in the Colab notebook. A measured local CPU fallback uses `--cpu`, 32 environments, 64-sample minibatches, 25-control unrolls, and four minibatches. A 12,800-transition benchmark reached 176.9 training transitions/second, with 79.7 seconds including compilation. The full 3,276,800-transition CPU continuation is now running, with periodic 16-episode evaluations, latest/best checkpoints, and final rollout rendering. Its initial randomized evaluation reported 1/16 successes under the lift-before-place requirement. Final independent verification will use 128 episodes. Its estimated duration is 5–6 hours; this is an estimate, not a completed result.

## 4. Evidence and remaining work

Delivered evidence is in the fork's [results folder](https://github.com/Trolleroof/simulation/tree/ml_onboarding/results):

- Expert video: `pick_place_rollout.mp4` (seed 42, 264 steps).
- BC demonstrations: `pick_place_dataset.pkl`; trained checkpoint: `best_policy.pth`; metrics: `results.json`; example: `task2_bc_rollout.mp4`.
- MJX expert physics video: `task3_expert_physics_rollout.mp4`.
- Rare learned-policy placement: `task3_policy_seed84.mp4`; corresponding checkpoint: `task3_policy_pickplace_5324800`.
- Matched trained/random evaluation: `task3_strict_evaluation.json` and `task3_strict_evaluation_random.json`; lift-before-release recheck: `task3_lift_then_release_recheck.json`.
- Preserved continuation checkpoint: `task3_policy_pickplace_6963200`.

The requested work is **not complete yet**. A rare released placement and a small observed improvement over random initialization are recorded. The CPU continuation must finish and its policy needs independent evaluation with randomized resets and the lift-before-place requirement. Colab's GPU usage limit prevents GPU continuation; the CPU fallback is running. This report records measured results and debugging experience without treating higher reward as proof of reliable placement.
