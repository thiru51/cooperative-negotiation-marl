# Cooperative negotiation at unsignalized intersections

Two cars arrive at a junction with no lights, no signs and no radio link between them. Only
one can go first. This repo trains them to sort that out by watching each other move.

The learner is MAPPO (multi-agent PPO), written from scratch here. Each car runs a particle
filter over the other car's *intent*, estimated from motion alone. The experiment the repo
is built around is a controlled A/B: the same learner, the same environment, the same
seeds, and one line of difference in the reward function.

---

## Status

Read this before anything else.

- The code is complete and tested. Environment, particle filter, MAPPO, reward variants,
  metrics, evaluation, and the entry-point scripts are all written and all import cleanly.
- The test suite passes: 55 tests, 8 files.
- The smoke test passes end to end, including the check that splitting the environments
  across worker processes reproduces the single-process trajectories exactly.
- **No training run has been executed yet.**
- **No performance numbers exist.** There are no results in this repo, no plots, no tables,
  no claims about deadlock rates. Nothing in this README quotes a number from a training
  run, because there has not been one.

The point of the repo in its current state is that it is *ready* to train. Section
[Training](#training-not-yet-run) has the exact commands, clearly marked as not yet run.

---

## Contents

- [The problem](#the-problem)
- [Why hand-coded priority rules fail](#why-hand-coded-priority-rules-fail)
- [How the approach works](#how-the-approach-works)
- [Repo layout](#repo-layout)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Check the GPU](#check-the-gpu)
- [Run the tests](#run-the-tests)
- [Smoke test](#smoke-test)
- [Training (not yet run)](#training-not-yet-run)
- [Evaluating a checkpoint](#evaluating-a-checkpoint)
- [Tuning for your machine](#tuning-for-your-machine)
- [Troubleshooting](#troubleshooting)
- [What this is not](#what-this-is-not)
- [References](#references)

---

## The problem

Picture a four-way junction with nothing controlling it. Two cars approach on crossing
paths and reach the line at about the same moment. There is one patch of tarmac in the
middle that both routes need, and it fits one car at a time.

Write down the four things that can happen when both drivers choose at once:

|                  | **B goes**       | **B yields**   |
|------------------|------------------|----------------|
| **A goes**       | crash            | fine: A first  |
| **A yields**     | fine: B first    | deadlock       |

Two of the four cells are fine. Both of the fine ones are *asymmetric* -- they require the
two cars to choose differently. Both of the bad ones are symmetric -- they happen when the
two cars choose the same thing.

That is the whole difficulty. When the two approaches are geometrically identical (same
distance out, same speed, same road), nothing in the situation distinguishes the two cars,
so nothing tells them which of the two good cells to aim for. If they reason identically
they land on the diagonal, and the diagonal is crash or deadlock.

Deadlock is the *safe* failure, which is why it is the one that actually happens. A cautious
policy has both cars sitting at the line waiting for the other to commit. It is not a crash,
so it does not look like a safety failure, but a vehicle frozen in a junction blocks
everything behind it and eventually provokes someone into a dangerous overtake. It is also
the failure that gets an autonomous vehicle programme cancelled, because it is embarrassing
and it is visible.

Humans clear this in about a second, without speaking. Someone lifts off the brake and rolls
forward six inches. The other driver sees the roll and holds. The negotiation is conducted
entirely in vehicle motion: **the movement is the message.** There is no channel other than
where the car is and how fast it is going.

## Why hand-coded priority rules fail

The obvious fix is a rule. "Yield to the right." "First to arrive goes first." "Lower
vehicle ID goes first." Every one of them breaks, and they break for different reasons that
are worth keeping separate.

**A rule needs a fact that does not exist.** "First to arrive" needs an unambiguous arrival
ordering. Two cars arriving within a few tenths of a second, seen through perception noise,
do not have one. Both onboard systems can honestly conclude they were first. Now both go.
Or both, being conservative, conclude the other was first, and both wait. The rule has
turned a coordination problem into a measurement problem and lost.

**A rule only works if the other party is running it too.** A cyclist, a pedestrian, a
delivery van whose driver has decided to go regardless, a car from somewhere with the
opposite convention. A rule-follower meeting a rule-breaker is either stuck forever or in a
collision, and the rule contains no instructions for either case.

**The safe rule produces the frozen robot.** "If in doubt, yield" is the right call for any
single encounter and a disaster over a day of driving. At a busy junction the doubt never
resolves and the car never moves.

**A tie-break is just the problem moved.** Lowest vehicle ID goes first works perfectly in
simulation and requires both cars to share a convention that does not exist on real roads.
It also degrades badly: it says what to do when both cars comply, and nothing at all about
what to do when the other car fails to concede.

**Rules cannot express a probe.** What a human actually does is not a decision, it is an
experiment: creep forward a little, watch the response, then either commit or back off. That
requires reasoning about how the other driver will react *to your action* -- a game, not a
lookup table. You cannot write that as a priority rule, because the right move depends on a
belief about the other party that only your own action can reveal.

This project takes the rule out entirely. Concretely: highway-env ships a `RegulatedRoad`
object with an explicit right-of-way rule that freezes the lower-priority vehicle at a
conflict point. The environment here rebuilds the road as a plain `Road` specifically to
delete that rule, so neither agent gets any scripted deference. See
`src/negotiation/envs/intersection.py`.

## How the approach works

### The Markov game

The encounter is modelled as a **two-agent general-sum Markov game**: a state space, one
action set per agent, transition dynamics, and *one reward function per agent*.

"General-sum" is the important word. It is not zero-sum -- the two cars are not competing
for a fixed prize, and a crash is terrible for both. It is also not fully cooperative -- the
two reward functions are not the same, because each car would rather be the one that goes
first. They agree completely on avoiding a crash and disagree on the ordering. That gap is
exactly what makes it a negotiation instead of a joint planning problem.

Both agents are **partially observing**. Neither can see the other's reward function,
policy, or chosen action. Each sees a noisy measurement of the other car's position,
heading and speed, plus the map, which both cars can see. To keep the learning problem
Markov, each agent's observation is augmented with the *belief* its own filter holds about
the other -- a belief-state MDP, which is what lets ordinary PPO machinery apply.

One agent's observation is 19 numbers (`OBS_DIM` in `envs/intersection.py`):

| block | size | contents |
|---|---|---|
| own kinematics | 4 | speed, distance to the conflict point, time to conflict, cleared flag |
| other's kinematics | 4 | same four, for the other car |
| interaction | 5 | difference in time-to-conflict, relative position (2), relative velocity (2) |
| belief | 2 | P(other car is assertive), and the binary entropy of that posterior |
| own history | 4 | one-hot of own last intent (3), plus a decaying average of how assertive it has been |

Everything is in the agent's own frame and normalised. The critic sees both agents'
observations concatenated, 38 numbers, self first.

### The action space: intention signalling

Three discrete actions, and they are not throttle values. Each is a **target speed** that
the car's low-level controller then tracks, with acceleration clipped to comfort limits
(`envs/intents.py`, `envs/vehicle.py`):

| action | target speed | reads as |
|---|---|---|
| `CREEP` | 2.0 m/s | "I'm interested, I haven't committed" |
| `YIELD_NUDGE` | 0.0 m/s | "after you" |
| `ASSERTIVE_ADVANCE` | 8.0 m/s | "I'm going" |

Two design points here, and both matter.

The action commands a *speed profile*, not an instantaneous jump, because acceleration is
clipped. So each intent takes roughly a second to express itself as motion, and each one
produces a different speed signature over that second. That signature is the only thing the
other car can read. If the intent set the speed instantly, `ASSERTIVE_ADVANCE` would be a
teleport and there would be nothing to infer from.

And the intent label is **never transmitted**. The other agent does not receive the action.
It only sees the motion that action produced, through noise. Calling them "signals" is about
what they are legible as, not about a message channel. There is none.

### Inferring the other car's intent

Each agent runs its own particle filter over the *other* agent (`inference/particle_filter.py`).
Nothing is shared between the two filters.

A particle filter approximates a probability distribution with a cloud of weighted guesses.
Each particle here carries:

- a pose and motion state: x, y, heading, speed, turn rate;
- a **binary behaviour hypothesis**: is this driver yielding, or assertive?
- two nuisance parameters: the driver's desired speed and its comfortable deceleration.

The loop each step is predict, reweight, resample.

**Predict.** With small probability (`switch_prob`, 0.02) the particle flips its hypothesis
-- drivers do change their minds, and without this the posterior saturates and can never
recover when the other car actually switches behaviour. Then each particle computes the
speed its hypothesised driver would be aiming for right now. An assertive driver holds its
desired speed. A yielding driver bleeds speed so as to stop a margin short of the conflict
point under comfortable braking, which is `v = sqrt(2 a d)` -- exactly the speed from which
that stop is still comfortable. The particle accelerates toward that target, clipped to its
own comfort limit, plus process noise, and its pose advances under a constant-turn-rate
model.

**Reweight.** Score every particle against the noisy measurement of the real car with a
Gaussian likelihood over position, heading and speed, in log space.

**Resample.** When the effective sample size falls below half the particle count, resample
systematically so the cloud does not degenerate to one particle.

The filter's output is `P(the other car is assertive)` plus the binary entropy of that
number, and both go into the observation -- the policy gets to know both what the belief is
and how confident it is.

Why a particle filter rather than a two-state Bayes filter over {yielding, assertive}?
Because the discrete hypothesis and the continuous nuisance parameters are coupled. You
cannot score "is this car yielding" without simultaneously estimating how hard *this
particular driver* brakes: a gentle braker and an assertive driver look identical for the
first half-second. The mixed discrete/continuous state is what a particle filter is for.

The measurement noise fed to the filter is drawn with the same standard deviations the
filter itself assumes. Handing it a noiseless state would make its measurement model wrong
in the flattering direction and produce a posterior that looks better than it is.

### The learner: MAPPO

`rl/mappo.py`, written out rather than imported from a library, so every choice is
inspectable. Centralised training, decentralised execution:

- **One actor, shared by both agents, with no agent-index input.** Withholding the index is
  deliberate and it is load-bearing. If the actor could condition on "am I agent 0", it
  could learn a fixed tie-break, and the junction would be solved by convention -- the
  hand-coded priority rule sneaking back in through the network. The only thing that can
  break the symmetry between the two agents is their observations, and the part of the
  observation that differs is the belief.
- **A centralised critic** that sees both agents' observations. Only used during training;
  it does not exist at execution time.

The PPO details are the four things Yu et al. (2022) found actually mattered for MAPPO:
clipped policy surrogate, clipped value loss, advantage normalisation per minibatch, and a
running value normaliser. Plus GAE, learning-rate annealing, gradient clipping, and bf16
mixed precision on the update.

### The Stackelberg reward, and why it breaks the deadlock

This is the core of the project, so it is worth doing slowly.

Both reward variants are pure functions of the same raw per-step quantities the environment
reports (`AgentTerms`): progress along the route, whether the car just cleared the junction,
collision, jerk, whether it is stationary, and which intent it signalled. Both share every
task term, at identical coefficients:

```
base = 0.30 * progress + 5.0 * just_cleared - 12.0 * collision - 0.004 * jerk - 0.02
```

**The symmetric variant** (the baseline, `configs/symmetric.yaml`) then charges both agents
identically: a penalty for standing still, and an equal penalty for signalling
`ASSERTIVE_ADVANCE`. Swap the two agents and the rewards swap with them. There is a test
that pins exactly this (`test_symmetric_reward_is_permutation_invariant`), because an
accidental asymmetry hidden in the baseline would confound the whole comparison.

Now here is why that baseline is expected to deadlock, and the argument is about the
*gradient*, not about the payoff table.

The actor is one shared network with no identity input. In a geometrically symmetric
encounter both agents feed it near-identical observations, so both sample from near-identical
distributions. A policy-gradient step changes that single network, which moves *both* agents
at once. In the 2x2 table at the top of this README, that means the learner can only travel
along the **diagonal**: (go, go) and (yield, yield). Those are precisely the two bad cells.
The two good cells are off-diagonal, and a shared symmetric policy in a symmetric state
cannot sit in an off-diagonal cell.

Of the two diagonal cells, one crashes and carries the largest penalty in the reward. So the
gradient runs away from mutual assertion, and what it runs toward is mutual yielding. Mutual
yielding is stable: any escape route runs back through the region where both cars advance
together and collide. The collision penalty walls the exit. The run settles into a standoff
and stays there.

**The Stackelberg variant** (`configs/default.yaml`) attacks that by making the reward
depend on a quantity that *differs between the two agents even when the geometry does not*:
the filter posteriors.

The two posteriors are not identical. Each agent's filter tracks a different car, with
independently sampled measurement noise and independently sampled process noise, and the two
intent histories diverge the moment either agent samples a different action. So `p_0` and
`p_1` -- how assertive each agent looks to the *other* one -- drift apart on their own. When
the gap exceeds `leader_margin` (0.15), the reward names the more assertive-looking agent
the **leader**:

- The **leader** stops paying the assertiveness penalty. Committing has become cheap for it.
- The **follower** keeps paying the assertiveness penalty, has its standing-still penalty cut
  to a fifth, and is paid a bonus for signalling `YIELD_NUDGE`. Conceding has become cheap
  for it.

Two agents in the same symmetric state now receive *different* rewards, and see *different*
observations (each sees its own posterior and its own entropy). The shared policy is no
longer stuck on the diagonal. What it can learn -- as a single symmetric function of the
observation, which is all it is allowed to be -- is:

> if my belief says the other car is the assertive one, concede; if my belief says it is
> hanging back, go.

A tiny random asymmetry in the belief gets amplified into a full role assignment, and the
role assignment is what the two good off-diagonal cells needed. That is the Stackelberg
structure: one player commits, the other best-responds to the commitment.

Three things to be honest about:

1. The leader label is computed from **both** filters, so it is centralised information. It
   is used only inside the reward, only during training. At execution each actor sees only
   its own filter's posterior. This is the same centralised-training / decentralised-execution
   split as the critic, applied to the reward as well.
2. When no leader is identified, or once either car has already cleared, the Stackelberg
   reward reduces to *exactly* the symmetric one. Two tests pin that. Shaping is for
   resolving a live conflict; leaving it on at an empty junction would pay an agent to keep
   signalling at nobody.
3. Nothing decides *which* agent becomes the leader. That is settled by whichever car
   happened to nudge first, through the belief. Deliberately so -- a fixed answer would be
   the hand-coded priority rule again.

`leader_margin` has to be tuned within a band: too small and the roles thrash on filter
noise every step, too large and the shaping never fires at all. `leader_switches_mean` is
logged for exactly this reason.

### What gets measured

Metrics are computed from the raw environment terms, **never** from the reward scalars, so
the two variants are scored on the same ruler (`metrics.py`):

| metric | meaning |
|---|---|
| `deadlock_rate` | fraction of episodes that timed out *and* had both cars essentially stationary for the last 2 seconds. The stall requirement separates a real standoff from "slow but still moving". |
| `resolve_rate` | fraction where both cars got through |
| `collision_rate` | fraction that ended in a crash |
| `time_to_resolve_mean` / `_p90` | seconds until both cars had cleared, over the episodes that resolved |
| `mean_jerk` | comfort proxy: mean absolute rate of change of acceleration |
| `min_separation_mean` | closest the two cars got |
| `leader_switches_mean` | how often the leader label flipped; a stability diagnostic for the shaping |

Evaluation always runs the same fixed suite of eight scenarios (`envs/scenarios.py`) so runs
are comparable. The first two, `sym_slow` and `sym_fast`, are the interesting ones: identical
approach distance and identical speed, so neither car has any geometric excuse to go first.

## Repo layout

```
src/negotiation/
  envs/
    intents.py          the three intents and their target speeds
    vehicle.py          IntentVehicle: highway-env vehicle whose speed target is an intent
    geometry.py         route -> arc-length frame; where two routes actually conflict
    intersection.py     the two-agent environment, observation features, raw per-step terms
    scenarios.py        the fixed 8-scenario evaluation suite + the training sampler
    belief_wrapper.py   runs one particle filter per agent, folds the posterior into the obs
    vec_env.py          SyncVecEnv (one process) and AsyncVecEnv (worker processes)
  inference/
    motion.py           CTRV motion model; what speed each hypothesised driver would want
    particle_filter.py  the intention filter
  rl/
    networks.py         shared actor, centralised critic, running value normaliser
    buffer.py           on-policy rollout storage, GAE, minibatching
    mappo.py            the MAPPO update
  rewards.py            SymmetricReward and StackelbergReward, from shared raw terms
  metrics.py            per-episode outcome tracking and aggregation
  evaluation.py         fixed-suite evaluation, scripted baseline policies, result table
  training.py           the training loop, logging, checkpointing, perf accounting
  device.py             device selection, AMP dtype, TF32, peak-VRAM accounting
  cli.py                shared CLI flags, so train.py and run_comparison.py cannot drift

scripts/
  check_gpu.py          doctor command: is this machine going to use the GPU?
  smoke_test.py         seconds-long end-to-end sanity check
  train.py              train one reward variant
  evaluate.py           score a checkpoint (or a scripted baseline) on the fixed suite
  run_comparison.py     the actual experiment: both variants, matched seeds and budgets

configs/
  default.yaml          the Stackelberg run
  symmetric.yaml        the baseline; differs from default.yaml on the `variant` line only

tests/                  pytest suite
Dockerfile              reproduces the pixi environment; CPU torch only
pixi.toml / pixi.lock   the exact, locked environment
requirements.txt        the plain-pip alternative
```

## Prerequisites

- Linux x86-64. Developed and tested on Ubuntu 24.04.
- **An NVIDIA driver, if you want the GPU.** Check it works before anything else:

  ```bash
  nvidia-smi
  ```

  You want a table with your GPU in it and a "CUDA Version" in the header. That number is
  the highest CUDA the *driver* supports, not what is installed. If `nvidia-smi` is not
  found, you have no driver and everything here will run on the CPU.

- **Disk.** The pixi environment with the CUDA-enabled PyTorch wheels is several GB. Give it
  10 GB of headroom:

  ```bash
  df -h .
  ```

- Python 3.11 if you are using the pip route. Pixi brings its own.
- A GPU is genuinely optional. Everything runs on the CPU, just slower -- and as it happens,
  the slow part of this project is the CPU physics, not the network. See
  [Tuning for your machine](#tuning-for-your-machine).

## Install

Two paths. Pick one. **(a)** is what the lockfile guarantees; **(b)** is for when you just
want a venv.

### (a) pixi (reproducible, recommended)

Pixi is a package manager that installs a project's exact environment from a lockfile, into
a `.pixi/` folder inside the project. Nothing global changes.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL -l          # put pixi on your PATH in this shell
pixi --version
```

Then, from the repo root:

```bash
cd cooperative-negotiation-marl
pixi install --locked
```

`--locked` means "install exactly what `pixi.lock` says, and fail rather than re-solve".
That is the whole point of committing the lockfile.

The first install is slow and downloads several GB -- PyTorch and the CUDA libraries it
bundles are most of it. On a home connection expect somewhere between several minutes and
half an hour, with no output until it is done. It is not stuck. If you want to watch it
work, from another terminal:

```bash
du -sh ~/.cache/rattler          # pixi's package cache, growing
```

Run things either through the named tasks in `pixi.toml`:

```bash
pixi run check-gpu
pixi run test
pixi run smoke
```

or drop into a shell with the environment active and use plain `python`:

```bash
pixi shell
python scripts/check_gpu.py
exit
```

Everywhere below, a bare `python scripts/...` assumes you are inside `pixi shell` or an
activated venv. Outside of one, prefix it with `pixi run`.

### (b) venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

That last line is not optional. `requirements.txt` lists the third-party dependencies;
`pip install -e .` installs *this* project as an importable package, in editable mode, so
`import negotiation` works from anywhere and picks up your edits without reinstalling. Skip
it and every script dies with `ModuleNotFoundError: No module named 'negotiation'`.

On Linux x86-64 the default PyPI `torch` wheel already bundles CUDA, so this gets you a GPU
build with no extra flags. If your driver is too old for it, install torch first from a
matching index and then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import negotiation, torch; print(torch.__version__, torch.cuda.is_available())"
```

## Check the GPU

Run this first on a fresh clone. If it says `cuda available  False`, nothing in the repo
will use the GPU and everything silently falls back to the CPU.

```bash
python scripts/check_gpu.py
```

Real output, from the machine this was developed on (a laptop RTX 4080):

```
torch                2.10.0+cu128
torch cuda build     12.8
cuda available       True
device count         1
device name          NVIDIA GeForce RTX 4080 Laptop GPU
compute capability   8.9
multiprocessors      58
total VRAM           11.57 GB
free VRAM            10.07 GB
bf16 supported       True
amp dtype chosen     torch.bfloat16
tf32 matmul          True
tf32 cudnn           True
cudnn benchmark      True
float32 precision    high

matmul benchmark, 4096x4096:
  fp32 (tf32 on)         21.2 TFLOP/s
  bf16                   47.3 TFLOP/s
  peak allocated     0.20 GB

GPU looks usable. Next: python scripts/smoke_test.py
```

Your numbers will differ. The lines that matter are `cuda available True` and an
`amp dtype chosen` that is not `None`. `bf16 supported True` means compute capability 8.0
or newer, and the code picks bfloat16 over float16 when it sees it -- bf16 has float32's
exponent range, so gradients cannot underflow and no loss scaler is needed.

The matmul benchmark is not part of the project -- it is there so you can tell in one
command whether the GPU is actually delivering, rather than sitting in a power-saving state
or being throttled. Skip it if you only want the device report:

```bash
python scripts/check_gpu.py --no-benchmark
python scripts/check_gpu.py --size 8192      # bigger matmul, if 4096 is too small to saturate
```

The script exits non-zero when there is no CUDA device, so it works in a CI check.

## Run the tests

```bash
pytest tests -q
```

or `pixi run test`. This exercises the geometry, the particle filter against synthetic
ground truth, both reward variants, the GAE and minibatching maths, the environment, the
worker-process split, and a tiny end-to-end training run on the CPU.

Result on the development machine:

```
.......................................................                  [100%]
55 passed in 33.76s
```

55 tests across 8 files. Most of the half-minute is the tests that actually step the
environment, which is the slow part of everything here.

A few useful subsets:

```bash
pytest tests -q -x                          # stop at the first failure
pytest tests/test_rewards.py -v             # just the reward logic
pytest tests/test_particle_filter.py -v     # just the intent inference
pytest tests -q -k "not worker"             # skip the multi-process tests
```

## Smoke test

An end-to-end check that takes seconds. It builds an environment, rolls a few steps under a
random policy, runs one tiny MAPPO update on whatever device is available, and verifies that
splitting the vector environment across worker processes reproduces the single-process
trajectories exactly.

```bash
python scripts/smoke_test.py
```

Output on the development machine, forced onto the CPU, in about five seconds:

```
cpu: cpu  amp=off
single env ok  obs_dim=19 state_dim=38 outcome=running posteriors=[0.0, 0.746]
vec+update ok  {'policy_loss': -0.0009, 'value_loss': 0.4516, 'entropy': 1.0985, 'approx_kl': 0.0001, 'clip_frac': 0.0, 'grad_norm': 0.6883}
  rollout 138 env-steps/s (8 envs, single process), peak VRAM None GB
worker split ok  4 envs in 2 processes reproduce the single-process rollout
smoke test passed
```

Reading that: 19 observation features per agent and 38 for the centralised critic, as
expected. The two posteriors differ, which means the filters are responding to the two
cars' different behaviour rather than sitting on the prior. Entropy near 1.0985 is
`ln(3)` -- a freshly initialised policy over three actions is still uniform, which is
correct at this point. `peak VRAM None` just means it ran on the CPU.

The `138 env-steps/s` is a real measurement of *this configuration*: 8 environments in a
single process on the CPU. It is not a training throughput. Training uses 64 environments
across worker processes, and the two do not scale linearly.

Options:

```bash
python scripts/smoke_test.py --device cpu       # force CPU
python scripts/smoke_test.py --skip-workers     # skip the multi-process reproducibility check
```

Run this before every training job. If the worker-split check fails, every number produced
with worker processes is measuring a different experiment from the single-process one, and
the run is worthless.

## Training (not yet run)

**None of the commands in this section have been run. There are no results. Every number
below is a configuration setting or an arithmetic consequence of one, not a measurement.**

### The Stackelberg run

```bash
python scripts/train.py --config configs/default.yaml
```

`configs/default.yaml` sets `variant: stackelberg`, 300,000 environment steps, seed 0, 64
parallel environments, rollout horizon 128, hidden width 128, and the leader/follower reward
coefficients. Output lands in `runs/stackelberg_seed0/`.

### The symmetric baseline

```bash
python scripts/train.py --config configs/symmetric.yaml
```

`configs/symmetric.yaml` is byte-identical to `default.yaml` apart from the `variant` line
and a comment. Same seed, same step budget, same everything else -- that is what makes the
comparison controlled. Output lands in `runs/symmetric_seed0/`.

If you edit one config, edit the other. Check they still differ on one line only:

```bash
diff configs/default.yaml configs/symmetric.yaml
```

### The actual experiment

The two runs above, paired, over several seeds, with the comparison table printed and the
full result written to JSON:

```bash
python scripts/run_comparison.py --total-steps 300000 --seeds 0 1 2 --tag v1
```

This trains six policies (2 variants x 3 seeds), evaluates each on the fixed eight-scenario
suite, and writes `results/v1.json` containing per-seed and per-scenario numbers for both
variants, plus the performance record of the first run.

Useful flags:

```bash
# quick shakedown -- confirms the plumbing end to end, far too short to learn anything
python scripts/run_comparison.py --total-steps 20000 --seeds 0 --tag shakedown

# more seeds, fewer eval episodes each
python scripts/run_comparison.py --total-steps 300000 --seeds 0 1 2 3 4 --eval-episodes 15 --tag v1_5seed

# periodic evaluation during training, so you can see when the deadlock breaks
python scripts/run_comparison.py --total-steps 300000 --seeds 0 --eval-every-updates 5 --tag v1_curve
```

### What a run actually costs

Arithmetic, not measurement. At the config defaults, one update collects
`horizon x num_envs = 128 x 64 = 8192` environment transitions, so a 300,000-step run is
`300000 // 8192 = 36` updates. Each transition is one decision by *each* of the two agents,
so that is roughly 600,000 agent decisions per run. `run_comparison.py` with three seeds is
six such runs.

**Wall clock: nobody has timed a full run, because nobody has done one.** What follows is an
extrapolation from a number that was measured, and it should be read as an order of
magnitude and nothing finer.

The only measurement that exists is the smoke test's: **138 environment-steps per second**
with 8 environments in a single CPU process, on the machine described above. Straight
division puts one 300,000-step run at `300000 / 138`, roughly half an hour, and the full
three-seed comparison at six times that.

Treat that as the pessimistic end. It was measured single-process on the CPU; a real run
uses 64 environments across several worker processes, and the environment step is the part
that parallelises. Whether that buys you 4x or 10x depends on your core count, and it has
not been measured. The point of the number is to tell you this is a "leave it running over
lunch" job, not a "leave it running over the weekend" one, and to tell you immediately if
something is badly wrong -- if the first few updates report tens of steps per second, stop
and look at `--num-workers`.

Run the smoke test on your own machine to get your own baseline before starting.

What is worth knowing in advance: **the bottleneck is the CPU, not the GPU.** Each
environment step runs highway-env physics plus two 256-particle filters; the networks are
two-layer 128-unit MLPs. `perf.json` at the end of every run records `rollout_fraction`,
the share of wall time spent collecting rather than updating, and it is expected to be the
large majority.

### What a run produces

Under `runs/<run-name>/`:

| file | contents |
|---|---|
| `config.json` | the fully resolved config, plus device, AMP dtype, worker count, observation dims |
| `train_log.jsonl` | one JSON object per logged update: env steps, throughput, mean step reward, PPO diagnostics (entropy, approximate KL, clip fraction, gradient norm), and a rolling window of outcome metrics |
| `checkpoint.pt` | actor, critic, value-normaliser state, config, dims |
| `perf.json` | device, throughput, rollout vs update split, peak VRAM allocated and reserved |
| `final_eval.json` | the fixed-suite evaluation of the final policy, per scenario and overall |
| `eval_during_training.jsonl` | only if `--eval-every-updates` was set |

The metrics inside `final_eval.json` and in the rolling window are the ones described in
[What gets measured](#what-gets-measured): deadlock rate, resolve rate, collision rate,
time-to-resolve (mean and p90), mean jerk, mean speed, minimum separation, leader switches.

`run_comparison.py` additionally writes `results/<tag>.json` with both variants side by side
and prints a delta table.

Live monitoring, once a run is going:

```bash
tail -f runs/stackelberg_seed0/train_log.jsonl
watch -n 5 nvidia-smi
```

## Evaluating a checkpoint

Scoring is separate from training, so a checkpoint can be re-scored at any time without
retraining.

```bash
python scripts/evaluate.py --checkpoint runs/stackelberg_seed0/checkpoint.pt --episodes 25
```

Write the full per-scenario result to JSON:

```bash
python scripts/evaluate.py \
  --checkpoint runs/stackelberg_seed0/checkpoint.pt \
  --episodes 25 \
  --out results/stackelberg_seed0_eval.json
```

Scripted reference policies, which need no checkpoint and are useful as anchors on the
metric scale:

```bash
python scripts/evaluate.py --policy always-yield   # the pathological standoff: deadlock anchor
python scripts/evaluate.py --policy always-go      # both cars commit: collision anchor
python scripts/evaluate.py --policy random         # random intents
```

`--policy always-yield` is worth running now, before any training, precisely because it is
the failure mode the whole project is about. It gives the deadlock metric a known
end-of-scale reading.

Other flags:

```bash
--stochastic          # sample actions instead of taking the argmax
--seed 12345          # evaluation seed; defaults to 12345 so runs are comparable
--device cuda         # defaults to cpu, which is usually fine for evaluation
--variant symmetric   # only affects the logged leader diagnostic and mean_return
```

## Tuning for your machine

### The honest version first

A bigger GPU will buy you very little here. Work out the rollout buffer at the default
config: `horizon x num_envs x agents = 128 x 64 x 2 = 16,384` rows, each holding a
19-float observation, a 38-float state, and about eight more floats for actions, log
probabilities, values, rewards, advantages and returns. At 4 bytes a float that is
`16384 x 65 x 4`, roughly four megabytes. The networks are two 128-unit hidden layers.
Nothing about this project stresses a modern GPU.

What actually limits throughput is the environment step: highway-env physics plus two
256-particle filters, per environment, on the CPU. So the knob that matters is **worker
processes**, and the ceiling is your core count.

```bash
nproc                                          # how many cores you have
python scripts/train.py --config configs/default.yaml --num-workers 8
```

`--num-workers 0` means "half the cores", which is the default and leaves headroom for the
main process feeding the GPU. `--num-workers 1` steps everything in-process, which is
simplest to debug and the right choice when something is behaving strangely.

### If you do have a bigger GPU

The way to use it is more parallel environments, not bigger networks -- more environments
means a wider batched policy forward per step, which is where the GPU helps, and it also
means more independent encounters per update.

```bash
# more environments, more workers to step them, bigger update batches
python scripts/train.py --config configs/default.yaml \
  --num-envs 256 --num-workers 16 --batch-size 8192
```

Keep the total step budget in mind: `num_envs` multiplies the transitions collected per
update, so at fixed `--total-steps` raising it means *fewer, larger* updates. If you raise
`--num-envs` by 4x, consider raising `--total-steps` too, or you have quietly cut the number
of gradient steps by 4x.

Other levers:

```bash
--batch-size 8192      # minibatch rows in the update; overrides --num-minibatches.
                       # This is the number that decides VRAM per backward pass.
--num-minibatches 4    # the alternative way to say the same thing
--horizon 256          # longer rollouts per update: better advantage estimates, more memory
--hidden 256           # wider MLPs. Try this last; it is unlikely to be the limitation
--epochs 8             # PPO passes over each rollout
--no-amp               # turn off bf16 mixed precision, e.g. to rule it out while debugging
--compile              # torch.compile the MLP bodies. Off by default: the warm-up costs
                       # tens of seconds and these networks are small enough that it rarely pays
```

### OOM: symptom and fix

The symptom is unmistakable:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ... GiB
(GPU 0; ... GiB total capacity; ... GiB already allocated; ... GiB free ...)
```

Fix, in order:

1. **Lower `--batch-size`.** It directly sets how many rows go through one backward pass.
   Halve it. This changes optimisation slightly but not the amount of data collected.
2. **Lower `--num-envs`, or `--horizon`.** Both shrink the rollout buffer, which is the only
   thing that grows with them.
3. **Lower `--hidden`.**
4. Check nothing else is on the GPU: `nvidia-smi`.

Every run writes its actual peak into `perf.json` as `peak_allocated_gb` (what the tensors
needed) and `peak_reserved_gb` (what the allocator held from the driver -- the number
`nvidia-smi` shows, and the one that OOMs you). Read those before guessing.

Given the arithmetic above, an OOM at anything near the default config almost certainly
means something *else* is occupying the GPU, not that this project needs the memory.

## Troubleshooting

**`ModuleNotFoundError: No module named 'negotiation'`**
You used the pip path and skipped `pip install -e .`. Run it from the repo root. On the pixi
path this cannot happen -- `pixi.toml` installs the project itself as an editable dependency.

**`ModuleNotFoundError: No module named 'torch'` inside the pixi environment**
`pixi install` was interrupted before it got to the PyPI dependencies (torch, gymnasium,
highway-env are all installed via pip inside the pixi environment, not from conda). Re-run
`pixi install --locked`.

**`cuda was requested but torch.cuda.is_available() is False`**
Raised deliberately by `device.py` rather than silently falling back to the CPU, so a run
you asked to be on the GPU never quietly takes ten times longer. Diagnose with:

```bash
python scripts/check_gpu.py
python -c "import torch; print(torch.version.cuda)"
```

`None` from that second command means you have a CPU-only wheel. Reinstall torch from a
CUDA index (see install path (b)). If `torch.version.cuda` is set but CUDA is still
unavailable, the driver is the problem: check `nvidia-smi`, and check
`CUDA_VISIBLE_DEVICES` is not set to an empty string.

**`pygame.error: No available video device`, or an SDL error, on a headless machine**
highway-env imports pygame, which wants a display even when it is only doing physics. Tell
SDL to use a dummy driver:

```bash
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
```

The Dockerfile already sets both.

**Training hangs at startup with `--num-workers` greater than 1**
Worker processes are started with `spawn` rather than `fork`, on purpose: forking a process
that has already initialised CUDA is a well-known way to get a hang that only appears on
someone else's machine. Spawn requires that the entry point is guarded by
`if __name__ == "__main__":`, which every script here is. If you wrote your own driver
script, add the guard. To rule workers out entirely:

```bash
python scripts/train.py --config configs/default.yaml --num-workers 1
```

**The worker-split check in the smoke test fails**
Stop and fix it before running anything. Environments are keyed by a *global* environment id
with per-environment RNGs precisely so that splitting them across processes is a speed change
and nothing else. If the trajectories diverge, multi-worker and single-worker runs are
different experiments and cannot be compared.

**A numpy 2.x error somewhere in highway-env**
numpy is pinned below 2 in both `pixi.toml` and `requirements.txt`. If you installed
something that dragged numpy 2 in, put it back:

```bash
pip install "numpy<2"
```

**The run is much slower than expected**
Look at `rollout_fraction` in `perf.json`. If it is close to 1, you are environment-bound:
raise `--num-workers` toward your core count. If it is low, you are update-bound, which at
this network size would be surprising -- check that something else is not competing for the
GPU.

**Rewards look wrong, or the leader never gets assigned**
`leader_switches_mean` in the logs is the diagnostic. Zero switches across a whole run means
`leader_margin` is too large and the shaping is never firing, so the Stackelberg variant has
silently degenerated into the symmetric one. Many switches per episode means it is too small
and the roles are thrashing on filter noise.

## What this is not

Stated plainly so nobody has to infer it.

- **There is no physical testbed.** No hardware, no scaled cars, no track, no real vehicle
  of any kind. Everything is simulation. Hardware is a possible future milestone and nothing
  more than that today.
- **There are no results.** No training run has been executed. See
  [Status](#status).
- **There is no vehicle-to-vehicle communication**, and adding it would defeat the purpose.
- **The other agent is not a human model.** It is the same learned policy. Whether any of
  this transfers to human drivers is untested and unclaimed.
- **Two vehicles, straight-through routes, one junction.** Nothing here has been tried at
  scale.
- The intents are named after human driving behaviours (`creep`, `yield-nudge`,
  `assertive-advance`) because that is what they are meant to evoke. The names are not a
  claim that the learned policy is human-like.

## References

The ideas this builds on, and what each one contributes here.

1. **Sadigh, Sastry, Seshia, Dragan (2016).** *Planning for Autonomous Cars that Leverage
   Effects on Human Actions.* Robotics: Science and Systems (RSS).
   The origin of the leader/follower framing used here: treat the interaction as a Stackelberg
   game in which the robot's action deliberately influences what the other driver does, rather
   than predicting the other driver as if they were weather.

2. **Fisac, Bronstein, Stefansson, Sadigh, Sastry, Dragan (2019).** *Hierarchical Game-Theoretic
   Planning for Autonomous Vehicles.* IEEE International Conference on Robotics and Automation
   (ICRA).
   Makes the game-theoretic interaction tractable by splitting it into a long-horizon strategic
   layer and a short-horizon tactical one. The reason this project's action space is three
   discrete intents rather than continuous throttle.

3. **Schwarting, Pierson, Alonso-Mora, Karaman, Rus (2019).** *Social behavior for autonomous
   vehicles.* Proceedings of the National Academy of Sciences (PNAS).
   Puts a scalar on how much a driver weighs others' welfare against their own (social value
   orientation) and estimates it online from observed motion. The direct ancestor of estimating
   a latent behavioural parameter from trajectory alone, which is what the particle filter here
   does with a binary yielding/assertive hypothesis.

4. **Tian, Li, Fujita, Zha (2019).** *Adaptive Game-Theoretic Decision Making for Autonomous
   Vehicle Control at Roundabouts.* Related work by these authors on level-k game-theoretic
   driver modelling appears in IEEE Transactions on Intelligent Transportation Systems (T-ITS).
   Level-k reasoning as an alternative to equilibrium assumptions, and the source of the idea
   that the *depth* of an opponent's reasoning is itself something to infer online.

5. **Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu (2022).** *The Surprising Effectiveness of PPO in
   Cooperative Multi-Agent Games.* Advances in Neural Information Processing Systems (NeurIPS),
   Datasets and Benchmarks Track.
   The MAPPO recipe implemented in `rl/mappo.py`: centralised critic, decentralised
   parameter-shared actors, and the specific implementation details (value normalisation,
   clipped value loss, advantage normalisation) that the paper found were what made it work.

No numbers from any of these papers are quoted anywhere in this repo.
