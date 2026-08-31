# Handoff notes

For whoever picks this up next, including me in three months. The README explains the idea.
This file is about the state of the code and the decisions behind it.

## Where things stand

The code is finished and tested. The experiment has not been run. There are no results,
no checkpoints, no plots. Everything below about training is a plan, not a report.

The single most useful thing you can do next is [start the runs](#what-to-run-first).

## What is implemented

Every module listed here is written, imports cleanly, and is covered by the test suite.

**Environment** (`src/negotiation/envs/`)

- `NegotiationIntersectionEnv` subclasses highway-env's `IntersectionEnv`. Two controlled
  vehicles on crossing straight-through routes. Background traffic is switched off, and the
  road object is rebuilt as a plain `Road` to delete highway-env's built-in right-of-way rule.
- Decisions happen at 5 Hz (`policy_frequency`), physics at 15 Hz, episodes cap at 20
  simulated seconds. So one episode is at most 100 decision steps.
- `IntentVehicle` takes over only the longitudinal command. highway-env keeps steering along
  the planned route. Acceleration is clipped to comfort limits, and per-step jerk is
  accumulated for the comfort metric.
- `geometry.py` flattens each route to a polyline once at reset and projects positions onto
  it, so "how far am I from the conflict point" is a scalar lookup instead of lane algebra
  every step. The conflict point is the closest approach between the two routes.
- The environment emits `AgentTerms` -- raw, reward-agnostic per-step quantities. Reward
  functions are pure functions of those. This is what makes the A/B controlled.
- `BeliefWrapper` runs one particle filter per agent and folds each agent's posterior into
  its own observation. Agent i's filter tracks agent 1-i, and nothing is shared between them.
- `vec_env.py` has `SyncVecEnv` (one process) and `AsyncVecEnv` (worker processes, `spawn`).
  Environments are keyed by global id with per-environment RNGs, so the worker split is a
  pure speed change; a test and the smoke test both assert the trajectories match exactly.

**Intent inference** (`src/negotiation/inference/`)

- `IntentionParticleFilter`: 256 particles, each carrying a pose/motion state, a binary
  yielding-or-assertive hypothesis, and two nuisance parameters (desired speed, comfortable
  deceleration). Predict / reweight / systematic-resample. Output is P(assertive) plus its
  binary entropy.
- `motion.py`: CTRV pose update with a first-order fallback as the turn rate goes to zero,
  and the target-speed model that says what each hypothesised driver type would be aiming for.
- The test suite drives a synthetic car with ground truth generated *outside* the filter and
  checks that the posterior separates the two behaviours. That is a real inference test, not
  a self-consistency check.

**Learner** (`src/negotiation/rl/`)

- `MAPPO`: shared actor with no agent-identity input, centralised critic over both agents'
  observations, clipped surrogate, clipped value loss, per-minibatch advantage normalisation,
  running value normaliser, GAE, LR annealing, gradient clipping, bf16 autocast.
- `RolloutBuffer` lives entirely on the training device, laid out `(T, envs, agents, ...)`.
  GAE runs in fp32 regardless of AMP.
- Checkpoints round-trip with or without `torch.compile`.

**Rewards, metrics, evaluation, training**

- `rewards.py`: `SymmetricReward` and `StackelbergReward`, sharing every task coefficient.
- `metrics.py`: per-episode outcome tracking, including the deadlock definition
  (timed out *and* stalled for the last two seconds).
- `evaluation.py`: fixed eight-scenario suite, scripted baseline policies
  (`always-yield`, `always-go`, `random`), result table formatting.
- `training.py`: the loop, JSONL logging, checkpointing, and a `perf.json` accounting of
  throughput, rollout-versus-update split, and peak VRAM.

**Entry points** (`scripts/`) — `check_gpu.py`, `smoke_test.py`, `train.py`, `evaluate.py`,
`run_comparison.py`. `train.py` and `run_comparison.py` share their runtime flags through
`cli.py` so they cannot drift apart.

## What to run first

In this order. Do not skip to the training run.

```bash
# 1. Is this machine going to use the GPU?
python scripts/check_gpu.py

# 2. Does everything still pass?
pytest tests -q

# 3. End-to-end plumbing, seconds
python scripts/smoke_test.py
```

Then anchor the metric scale *before* training anything, so there is a reference point for
what a total deadlock and a total pile-up look like on this evaluation suite:

```bash
python scripts/evaluate.py --policy always-yield --out results/anchor_always_yield.json
python scripts/evaluate.py --policy always-go    --out results/anchor_always_go.json
python scripts/evaluate.py --policy random       --out results/anchor_random.json
```

Then a deliberately too-short paired run, purely to confirm the comparison pipeline writes
what it should. This will not learn anything and its numbers mean nothing:

```bash
python scripts/run_comparison.py --total-steps 20000 --seeds 0 --tag shakedown
cat results/shakedown.json | head -40
```

Only then the real thing:

```bash
python scripts/run_comparison.py --total-steps 300000 --seeds 0 1 2 --tag v1
```

While it runs, watch `resolve`, `deadlock` and `ent` (policy entropy) in the printed log.
The two things to look for early:

- **Entropy collapsing to near zero in the first few updates** means the policy has committed
  to a single intent before it has learned anything. `entropy_coef` is already set higher than
  a typical single-agent PPO run for this reason; raise it further if it still happens.
- **`leader_switches_mean` at zero for a whole run** means `leader_margin` is too large and the
  Stackelberg shaping never fired, so the treatment condition silently became the control.
  That would invalidate the comparison, not just weaken it. Check it early.

## Known issues and open items

Nothing here is a crash. These are the things I would want to know about.

- **The whole experiment is unrun.** Every claim in the README about what the symmetric
  baseline does is an argument, not an observation. It is entirely possible the baseline does
  not deadlock as cleanly as predicted, in which case there is no effect to demonstrate and
  the premise needs revisiting. This is the main risk and it can only be settled by running it.

- **`leader_margin` is unvalidated.** 0.15 was chosen by reasoning about filter noise, not by
  a sweep. It sits between two failure modes: too small and the leader label thrashes every
  step on noise, too large and the shaping never fires. Diagnosing it needs `leader_switches_mean`
  from a real run, which is why it is not settled yet.

- **The reward coefficients are unvalidated too.** They are chosen so that the terms sit at
  plausible relative magnitudes over an episode, not tuned. Expect to revisit
  `assertive`, `yield_bonus` and `follower_wait_discount` after the first run.

- **`entropy_coef` at 0.02 is a guess in the same category** -- higher than usual on purpose,
  because with three actions and a narrow good region the policy collapses fast, but nobody
  has watched it happen yet.

- **Episode length versus deadlock detection.** An episode times out at 20 simulated seconds
  and the deadlock test looks at the last 2 seconds of it. If a policy learns to stall for 18
  seconds and then cross, that counts as resolved, and the time-to-resolve metric is what
  exposes it. Read `time_to_resolve_p90`, not just the mean.

- **The evaluation suite is small.** Eight fixed scenarios, and only two of them are the
  head-on symmetric case that the project is really about. Fine for a controlled comparison,
  thin for a general claim.

- **`AgentTerms.speed`, `distance_to_conflict` and `time_to_conflict` are recorded but not
  used by either reward.** They are there for diagnostics and for future reward variants.
  Not dead code exactly, but not load-bearing either.

- **`_compute_terms` computes `dt` and discards it** (`_ = dt`). Harmless leftover; the
  jerk normalisation moved into `IntentVehicle.step`.

- **The pixi environment has not been rebuilt from the lockfile recently.** The last full
  verification pass -- all modules importing, 55 tests passing, the smoke test end to end --
  was done in a plain venv built from `requirements.txt`, on CPU-only PyTorch. `pixi.lock`
  itself has not changed, so there is no reason to expect it to fail, but nobody has watched
  `pixi install --locked` finish on a clean machine lately. Worth confirming before you
  depend on it for a reproducibility claim.

- **The GPU path has been checked, but not the GPU training path.** `scripts/check_gpu.py`
  reports a working CUDA device with bfloat16 on the development machine. Nothing has been
  trained on it.

- **No physical testbed exists** and none is planned for v1. See `END_GOAL.md`.

## Key design decisions, and why

The decisions that would be easy to undo by accident.

**The road has no right-of-way rule.** highway-env's `RegulatedRoad` freezes the
lower-priority vehicle at a conflict point. That is exactly the hand-coded priority rule the
project exists to do without, so `_make_road` rebuilds the road as a plain `Road`. Putting
`RegulatedRoad` back would make the environment solve the problem for the agents and the
result would be meaningless.

**The actor gets no agent index.** One shared network, and the only thing that can break the
symmetry between the two agents is their observations. Give it an index and it can learn a
fixed tie-break -- the priority rule, sneaking back in through the network weights. A test
pins the actor's input width to exactly one agent's observation for this reason.

**The action space is three intents, not throttle.** Each intent is a target speed the
low-level controller tracks under clipped acceleration, so each takes about a second to
express itself as a distinct speed signature. That signature is the only channel between the
two cars. Instantaneous speed changes would leave the filter nothing to infer from, and
continuous throttle would make every action a blend rather than a legible claim.

**The intent is never transmitted.** The other agent sees only the motion, through noise.
Adding a message channel would make the inference problem disappear, and with it the project.

**Both reward variants are pure functions of the same raw terms.** The environment computes
`AgentTerms` and knows nothing about rewards. Dynamics and measured quantities are identical
across the two conditions; only the mapping to a scalar differs. Metrics are computed from
the raw terms too, never from reward scalars, so the two variants are scored on the same
ruler and the better-shaped one does not get a better-looking score for free.

**The Stackelberg reward reduces exactly to the symmetric one when no leader is identified,
and once either car has cleared.** Two tests pin this. Shaping is for resolving a live
conflict; leaving it on at an empty junction would pay an agent to keep signalling at nobody.

**The leader label uses both filters, and is training-time only.** It is centralised
information used inside the reward. At execution each actor sees only its own filter's
posterior. Same centralised-training / decentralised-execution split as the critic, applied
to the reward as well. If you ever feed the leader label into the observation, you have
broken decentralised execution.

**Nothing decides which agent becomes the leader.** The role falls out of the beliefs, which
fall out of which car happened to nudge first. A fixed answer would be the priority rule
again.

**The filter is fed noise with the same sigmas it assumes.** Handing it a noiseless state
would make its measurement model wrong in the flattering direction and produce a posterior
that looks better than it is.

**The filter has a per-step probability of changing its mind** (`switch_prob`). Without it
the posterior saturates and can never recover when the other car genuinely switches
behaviour, which is precisely the case the project cares about.

**Environments are keyed by a global id with per-environment RNGs.** This is what makes
splitting them across worker processes a pure speed change. A shared sampler RNG would make
results depend on how the environments happened to be partitioned, and every multi-worker
number would be incomparable with every single-worker one. The smoke test checks it.

**Workers are started with `spawn`, not `fork`.** The parent has usually initialised CUDA by
then, and forking a live CUDA context is a well-known way to get a hang that only appears on
someone else's machine.

**Logits are cast back to fp32 before the softmax.** Everything downstream -- the log-ratio,
the clip test, the entropy bonus -- compares small differences between numbers near 1, and
bf16 does not carry enough digits for that.

**GAE runs in fp32 regardless of AMP.** It is a length-T recursion, so a rounding error made
at the last timestep is still sitting in the estimate at the first.

**MAPPO is written out rather than imported.** Slower to write, but every choice is
inspectable, and this repo exists partly to be explained line by line.
