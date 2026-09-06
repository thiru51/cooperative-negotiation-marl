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

## v2: the same experiment with more exploration

`entropy_coef` raised from 0.02 to 0.05 in **both** variants. Everything else identical:
same 300,000 steps, same seeds 0/1/2, same evaluation suite.

A first attempt at this was a **no-op and is not reported**: the setting was changed in
`configs/*.yaml`, but `run_comparison.py` builds its config from CLI flags and never reads
those files, so it re-ran v1 exactly. The identical output to three decimals is what gave
it away. `--entropy-coef` is now a real command-line flag; v2 below was run with it.

| metric | stackelberg | symmetric | delta |
|---|---|---|---|
| deadlock_rate | 0.000 | 0.000 | +0.000 |
| resolve_rate | **0.777** | 0.417 | +0.360 |
| collision_rate | 0.183 | 0.000 | +0.183 |
| time_to_resolve_mean | 12.49 | 17.49 | -5.01 |
| mean_speed | 6.19 | 5.39 | +0.81 |
| mean_jerk | 1.205 | 0.532 | +0.673 |

Exploration was the binding constraint. Resolve went from 0.203 to 0.777 for Stackelberg
and 0.125 to 0.417 for symmetric, from a single hyperparameter change. v1's ceiling was
not the reward design; it was the policy committing before it had learned anything.

### The per-seed picture, which is the interesting part

| variant | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| stackelberg resolve | 0.855 | 0.765 | 0.710 | 0.777 |
| stackelberg collision | 0.145 | 0.235 | 0.170 | 0.183 |
| symmetric resolve | 0.125 | **1.000** | 0.125 | 0.417 |
| symmetric collision | 0.000 | 0.000 | 0.000 | 0.000 |

**Stackelberg is consistent.** All three seeds land between 0.71 and 0.855. That is a
real effect, unlike v1 where the mean rested on one seed.

**Symmetric is bimodal.** Two seeds collapse to the same degenerate 0.125 policy. One
seed found something better than anything Stackelberg produced: **1.000 resolve with
0.000 collisions** at 12.08 s. So the symmetric reward has the higher ceiling; it just
reaches it one time in three.

What the Stackelberg shaping buys is therefore **reliability, not peak performance**.
That is a narrower claim than the one this project started with, and it is the one the
data supports.

**The collision cost is real, and v3 shows it does not tune away.** Every Stackelberg
seed collides in 14.5% to 23.5% of episodes here. Doubling the collision penalty does
not help (0.198 across six seeds); quadrupling it reaches zero collisions only by
dropping resolve to 0.102. See the v3 section above.

`leader_switches_mean` was 2.08-2.65 across the Stackelberg runs, so the leader/follower
signal fired throughout and the comparison is valid.

## Verifying the results

Every number above can be re-derived from what is in this repository, without retraining.
The trained weights are committed (`runs/<tag>_<variant>_seed<N>/checkpoint.pt`, 170 KB
each) alongside the per-update training logs.

To reproduce the headline v2 Stackelberg seed-0 row:

```bash
python scripts/evaluate.py \
  --checkpoint runs/v2_stackelberg_seed0/checkpoint.pt \
  --variant stackelberg --seed 9000 --episodes 25 \
  --out /tmp/verify.json
```

This returns `resolve 0.8550, collision 0.1450, deadlock 0.0000` -- identical to the
`stackelberg` seed-0 entry in `results/v2.json`.

**The `--seed 9000` matters and is easy to miss.** Training scores its final evaluation
with `seed = cfg.seed + 9000` (`src/negotiation/training.py`), while `scripts/evaluate.py`
defaults to `--seed 12345`. Evaluate the same checkpoint under the default seed and you get
0.8625 rather than 0.8550 -- a different sample of the scenario suite, not a different
policy. Use `training seed + 9000` to match a reported number.

**One known discrepancy.** The three outcome rates reproduce exactly. `time_to_resolve_mean`
comes back at 11.95 s against the recorded 11.99 s, a 0.3% gap, because the two paths
aggregate it slightly differently (mean over per-scenario means vs mean over resolved
episodes). The rates are the metrics the conclusions rest on; the timing figure should be
read as approximate to about a tenth of a second.

**What is not committed:** the MetaDrive/highway-env episode caches and the optimiser state.
Neither is needed to re-score a checkpoint.

## v3: is the collision cost tunable away?

v2 left one thing flagged as "the first thing to fix": Stackelberg resolves 0.777 of
encounters but collides in 18.3% of them, and the symmetric baseline collides in none.
The obvious question is whether that is just an under-weighted collision term. The
reward has a `collision` coefficient, default 12.0, so sweep it.

**A prerequisite bug.** The coefficient was unreachable. `run_comparison.py` builds its
config from CLI flags and never reads `configs/*.yaml`, and no flag exposed the reward
weights -- so a sweep over `collision` would silently have re-run the identical
experiment three times, exactly as the first `entropy_coef` sweep did. `REWARD_FIELDS`
and `--collision-penalty` now exist, and `tests/test_reward_flags.py` pins every
declared coefficient to a working flag.

### The frontier

Penalties 12 / 24 / 48, both variants, 300,000 steps, `entropy_coef` 0.05. Penalty 24
was re-run with three further seeds once its first result looked too good, so that row
is six seeds and the others are three.

| penalty | variant | resolve | collision | time to resolve |
|---|---|---|---|---|
| 12 | stackelberg | 0.777 | 0.183 | 12.49 |
| 12 | symmetric | 0.417 | 0.000 | 17.49 |
| **24** | stackelberg | 0.586 +-0.202 | 0.198 +-0.163 | 12.10 |
| **24** | symmetric | 0.562 +-0.438 | 0.000 | 13.44 |
| 48 | stackelberg | 0.102 | 0.000 | 16.83 |
| 48 | symmetric | 0.417 | 0.000 | 18.08 |

**The collision cost does not tune away.** Doubling the penalty leaves Stackelberg's
collision rate where it was -- 0.183 at penalty 12, 0.198 across six seeds at 24. Only
at penalty 48 does it reach zero, and there resolve collapses to **0.102**: the policy
buys safety by refusing to cross. There is no setting in this range where the
Stackelberg reward is both safe and effective.

### A result I nearly reported, and why I did not

The first three seeds at penalty 24 gave symmetric **1.000 resolve with 0.000
collisions**, on every one of them. Read alone that overturns this project's thesis:
give the symmetric reward a properly weighted penalty and the asymmetric shaping is not
merely unnecessary but harmful.

Three more seeds at the same setting returned **0.125, 0.125, 0.125** -- the degenerate
policy. Per-seed, symmetric at penalty 24 is:

    1.000  1.000  1.000  0.125  0.125  0.125

**Bimodal, with nothing in between.** The first three seeds were one mode of a
two-mode distribution, and three seeds could not have told the difference. The
apparent refutation was sampling.

What six seeds actually show is the v2 claim again, with more evidence behind it and at
a different penalty: **symmetric is bimodal (sd 0.438), Stackelberg is comparatively
consistent (sd 0.202), and Stackelberg pays for that consistency in collisions.** Mean
resolve is now nearly identical between the two (0.586 against 0.562), so the honest
summary is narrower than v2's: what the asymmetric shaping buys is *reliability*, and
at penalty 24 it no longer buys much mean performance.

### What still is not established

**Six seeds is still few for a bimodal distribution.** Three-and-three is consistent
with anything from a 30% to a 70% chance of the good mode. Estimating that rate is what
the next run should do, not another penalty value.

**No mechanism.** Nothing here explains *why* symmetric lands in one mode or the other,
and the per-seed split is clean enough that something identifiable is probably going on
early in training. The per-update logs are committed and nobody has looked.

---

## Twelve seeds: the v2 claim does not survive at penalty 24

The section above ended by saying six seeds could not pin down how often the symmetric
reward finds its good mode, and that estimating that rate was what the next run should
do. Six more seeds were run at penalty 24. Twelve in total:

| | resolve | collision | good mode |
|---|---|---|---|
| stackelberg | 0.489 +-0.287 | **0.245** | -- |
| symmetric | 0.490 +-0.431 | **0.000** | **5/12 (42%)** |

Per-seed resolve:

    stackelberg  0.73 0.50 0.69 0.26 0.88 0.48 0.13 0.78 0.13 0.52 0.00 0.81
    symmetric    1.00 1.00 1.00 0.13 0.13 0.13 0.13 1.00 0.13 0.13 0.13 1.00

**Mean resolve is now identical -- 0.489 against 0.490 -- and the symmetric reward never
collides.** On this evidence, at this penalty, the symmetric baseline is simply better:
the same success rate for none of the collisions.

### What that does to the v2 claim

v2 concluded that asymmetric shaping buys **reliability**, trading a worse collision rate
for consistency across seeds. At penalty 12 with three seeds that was what the data said.
At penalty 24 with twelve seeds it is much weaker:

- The consistency gap narrowed. Stackelberg's sd is 0.287 against symmetric's 0.431 --
  still lower, but Stackelberg now has a **0.00 seed** and two at 0.13. It is not
  reliable; it is merely less bimodal.
- The cost did not narrow. 24.5% collisions against zero.
- And the mean advantage that justified the trade has gone entirely.

So the claim has to be narrowed again, and this is the third time this project's premise
has moved under measurement: the deadlock premise did not reproduce, the collision cost
did not tune away, and now the reliability advantage does not survive twelve seeds at a
higher penalty.

**What is still true:** the symmetric reward is genuinely bimodal, landing in a perfect
policy (1.000 resolve, 0.000 collisions) in **42% of seeds** and a degenerate one (0.125)
in the rest, with nothing between. That bimodality is real, reproducible and the most
interesting thing here. A practitioner would take the symmetric reward and restart it
until it lands -- which costs three runs on average and beats anything the Stackelberg
shaping produced.

**What would be worth doing:** the per-update logs for all twelve seeds are committed and
nobody has looked at them. The split is clean enough that something identifiable almost
certainly happens early in training, and predicting the mode from the first few updates
would be worth more than another reward variant.

---

## Honest summary

**v1 (`entropy_coef` 0.02).** Neither variant learns to resolve reliably. Entropy
collapses, resolve peaks at 0.40 around update 13 and decays to 0.07. The Stackelberg
mean of 0.203 rests on one seed of three.

**v2 (`entropy_coef` 0.05).** Both variants improve sharply. Stackelberg resolves 0.777
of episodes consistently across seeds, against 0.417 for symmetric, and resolves 5 s
faster. It pays for this with an 18.3% collision rate that the symmetric baseline does
not have. The symmetric baseline is bimodal and its best seed (1.000 resolve, 0.000
collisions) beats every Stackelberg seed.

**The original premise remains unsupported.** This project was built on the claim that a
symmetric reward collapses into wait-wait deadlock and Stackelberg shaping breaks it. The
symmetric baseline deadlocked in **0.000** of episodes in both v1 and v2. The scripted
`always-yield` anchor deadlocks 1.000 of the time, so the environment and the metric both
work -- a *learned* symmetric policy simply does not go there. It creeps to the timeout
instead, which dodges the strict deadlock test (stalled for the final two seconds).

The defensible claim from this data is: **Stackelberg-style asymmetric shaping makes
negotiation outcomes consistent across seeds where a symmetric reward is bimodal, at the
cost of a materially worse collision rate.** The deadlock claim needs either restating in
terms of timeouts, or dropping.
