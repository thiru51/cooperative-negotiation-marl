# Results

First real experiment, run 2 September 2026. Everything here comes from
`results/*.json`, written by the scripts in this repo. Nothing is estimated.

## What was run

| | |
|---|---|
| Steps per run | 300,000 environment steps |
| Seeds | 0, 1, 2 |
| Variants | `stackelberg`, `symmetric` |
| Runs | 6 (2 variants x 3 seeds) |
| Evaluation | 8 fixed scenarios, 25 episodes each, argmax actions |
| Hardware | RTX 4080 Laptop, 12 GB, bf16 autocast |
| Throughput | 886.7 env-steps/s at 64 envs (rollout 99%, update 1%) |
| Wall clock | 332.6 s per run |

`entropy_coef` was 0.02 for this run (`v1`). Everything except the reward function was
held fixed across the two variants: same environment, same scenario sampler, same seeds,
same step budget.

## Anchors

Scripted policies, scored on the same evaluation suite, to fix the scale. Without these
the trained numbers mean nothing.

| policy | resolve | deadlock | collision | time to resolve | jerk |
|---|---|---|---|---|---|
| always-yield | 0.000 | **1.000** | 0.000 | n/a | 0.50 |
| always-go | 0.120 | 0.000 | **0.880** | 6.80 | 0.90 |
| random | 0.030 | 0.000 | 0.360 | 19.72 | 16.77 |

Total politeness deadlocks every episode. Total aggression collides in 88% of them. So
the environment can produce both failure modes, and a useful policy has to beat both.

## The paired comparison (v1)

| metric | stackelberg | symmetric | delta |
|---|---|---|---|
| deadlock_rate | 0.000 | 0.000 | +0.000 |
| resolve_rate | 0.203 | 0.125 | +0.078 |
| collision_rate | 0.088 | 0.000 | +0.088 |
| time_to_resolve_mean | 12.48 | 20.20 | -7.73 |
| mean_speed | 5.44 | 5.03 | +0.41 |
| mean_jerk | 0.601 | 0.349 | +0.252 |

`leader_switches_mean` was ~2.0 in both variants, so the leader/follower signal fired
throughout and `leader_margin = 0.15` is not too large. That was the check that had to
pass for the comparison to mean anything at all.

## What this does not show

**The premise did not reproduce.** The argument for the Stackelberg reward is that a
symmetric reward collapses into a wait-wait deadlock. The symmetric baseline deadlocked
in **0.000** of evaluation episodes. There was no deadlock to break, so this run cannot
support the deadlock-breaking claim. That is the headline finding and it is negative.

**The mean hides the variance.** Per-seed resolve rates:

| variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| stackelberg | 0.000 | 0.000 | **0.610** |
| symmetric | 0.125 | 0.125 | 0.125 |

The Stackelberg average of 0.203 is one seed out of three. The other two produced
policies that never resolve a single episode. An effect carried by one seed in three is
not an effect yet. The symmetric numbers being identical to three decimal places across
seeds says its policies converged to the same degenerate behaviour every time.

**The collision cost is real.** Where Stackelberg does resolve (seed 2), it also collides
in 26.5% of episodes. Faster resolution bought with collisions is not a good trade for
this problem.

## Why it turned out this way

Entropy collapses. Policy entropy starts at 1.09 (uniform over three actions) and falls
to about 0.08 by the end of training. Resolve rate tracks it and then reverses:

| update | entropy | resolve | collision |
|---|---|---|---|
| 13 | 0.286 | **0.400** | 0.09 |
| 15 | 0.189 | 0.330 | 0.03 |
| 17 | 0.141 | 0.270 | 0.00 |
| 20 | 0.112 | 0.180 | 0.00 |
| 24 | 0.080 | 0.070 | 0.00 |

The policy finds a reasonable solution around update 13, then over-commits to never
colliding and loses the ability to cross at all. It ends up creeping until the 20-second
timeout, which avoids collision, avoids the strict deadlock test (that needs the agent
stalled for the last two seconds, and a creeping agent is not stalled), and resolves
nothing.

`configs/default.yaml` predicted exactly this in a comment written before the run:

> Higher than a typical single-agent PPO run on purpose. With only three actions and a
> reward whose good region is narrow, the policy collapses onto one intent within a few
> updates unless exploration is held open.

0.02 was not high enough.

## What is running next

`v2`: the same 6-run protocol with `entropy_coef` raised from 0.02 to 0.05 in **both**
config files. Raising it in only the Stackelberg config would confound the comparison.
Results will be appended here when it finishes.

## Honest summary

At 300,000 steps with `entropy_coef = 0.02`, neither reward variant learns to resolve
this intersection reliably. The Stackelberg variant resolves more often on average and
much faster when it resolves at all, but that average rests on one seed of three and
carries a collision rate the symmetric baseline does not have. The deadlock-breaking
claim is untested, because the symmetric baseline never deadlocked.
