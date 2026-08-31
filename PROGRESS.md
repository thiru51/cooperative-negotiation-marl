# Progress

State of the project as of the last commit. Boxes are ticked only for things that exist and
have been run. Nothing here is ticked on the strength of intending to do it.

**Summary: the code is done, the experiment is not.** No training run has been executed and
no results exist.

## Done

### Environment
- [x] Two-agent unsignalized intersection built on highway-env
- [x] highway-env's built-in right-of-way rule removed (plain `Road`, not `RegulatedRoad`)
- [x] Background traffic switched off; clean two-body encounter
- [x] Three intention-signalling actions (creep, yield-nudge, assertive-advance) as tracked
      target speeds under clipped acceleration
- [x] Arc-length route frames and conflict-point geometry
- [x] Raw, reward-agnostic per-step terms (`AgentTerms`) so both reward variants read the
      same quantities
- [x] Fixed eight-scenario evaluation suite plus a randomised training sampler
- [x] Per-agent observation: own kinematics, other's kinematics, interaction terms, belief,
      own intent history
- [x] Centralised state for the critic (both observations, self first)

### Intent inference
- [x] Particle filter over a binary yielding/assertive hypothesis plus latent driver
      parameters
- [x] CTRV motion model with a stable near-zero-turn-rate branch
- [x] Systematic resampling on effective-sample-size collapse
- [x] Posterior and its entropy folded into the observation
- [x] One independent filter per agent, no sharing, no vehicle-to-vehicle channel
- [x] Tested against synthetic ground truth generated outside the filter

### Learner
- [x] MAPPO written from scratch: shared actor with no agent-identity input, centralised critic
- [x] Clipped policy surrogate, clipped value loss, per-minibatch advantage normalisation,
      running value normaliser
- [x] GAE with correct handling of both termination and time-limit truncation
- [x] On-device rollout buffer, minibatching, LR annealing, gradient clipping
- [x] bf16 mixed precision with fp32 logits and fp32 GAE
- [x] Checkpoint save/load, round-tripping with and without `torch.compile`

### Rewards
- [x] `SymmetricReward` (control), permutation-invariant by construction and by test
- [x] `StackelbergReward` (treatment): leader from the two posteriors, follower paid to concede
- [x] Both variants share every task coefficient; only the signalling shaping differs
- [x] Shaping switches off when no leader is identified and once either car has cleared

### Infrastructure
- [x] Vector environment: single-process and worker-process versions
- [x] Worker split verified to reproduce single-process trajectories exactly
- [x] Device setup: auto-detect, AMP dtype choice, TF32, peak-VRAM accounting
- [x] `scripts/check_gpu.py` doctor command
- [x] `scripts/smoke_test.py` end-to-end sanity check
- [x] `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_comparison.py`
- [x] Shared CLI flags between `train.py` and `run_comparison.py`
- [x] Two configs differing on one line (`configs/default.yaml`, `configs/symmetric.yaml`)
- [x] Outcome metrics: deadlock rate, resolve rate, collision rate, time-to-resolve mean and
      p90, jerk, mean speed, minimum separation, leader switches
- [x] Scripted baseline policies (always-yield, always-go, random) as metric anchors
- [x] Per-run artifacts: config, JSONL training log, checkpoint, perf record, final evaluation
- [x] `pixi.toml` + `pixi.lock` for a reproducible environment
- [x] `requirements.txt` for a plain venv
- [x] Dockerfile
- [x] 55 tests across 8 files, all passing
- [x] README, HANDOFF, END_GOAL, this file

## Not done

The whole experimental half of the project.

- [ ] **Run the comparison.** `scripts/run_comparison.py`, both variants, matched seeds and
      step budget. This is the deliverable and it has not been started. Nothing else in this
      list can happen before it.
- [ ] Anchor the metric scale with the scripted policies (`--policy always-yield`,
      `always-go`, `random`) before training, so there is a reference reading
- [ ] A short too-short shakedown run to confirm the comparison pipeline writes what it should
- [ ] Repeat over at least three seeds
- [ ] Write `results/v1.json` and commit it
- [ ] Results section in the README, with the exact command and the commit hash
- [ ] Plot deadlock rate and time-to-resolve for both variants across the eight scenarios
- [ ] Honest write-up of what the result does not show

## Not done, and needs a run before it can be

These are open questions that cannot be answered by reading the code.

- [ ] Validate `leader_margin` (0.15). Chosen by reasoning about filter noise, not by a sweep.
      `leader_switches_mean` from a real run decides it.
- [ ] Validate the reward coefficients. Set to plausible relative magnitudes, not tuned.
- [ ] Confirm `entropy_coef` (0.02) actually keeps exploration open long enough
- [ ] Measure real wall-clock and throughput for a full run, and the rollout-versus-update split
- [ ] Check whether the symmetric baseline deadlocks as cleanly as the argument predicts. If
      it does not, the premise needs revisiting.

## Out of scope for v1

Listed so they are not mistaken for things that were forgotten.

- [ ] More than two vehicles
- [ ] Turning routes
- [ ] Pedestrians, cyclists, background traffic
- [ ] Human driver models
- [ ] Continuous throttle
- [ ] Vehicle-to-vehicle messaging
- [ ] **Any physical testbed.** No hardware exists. A scaled testbed is a possible future
      milestone and nothing more than that today.
