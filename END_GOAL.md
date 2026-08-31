# End goal

What "finished" means for version 1 of this project. Written down so that it is obvious
when to stop, and so that a bad result still counts as a result.

## The one-sentence goal

Show that two self-interested cars, meeting at an intersection with no lights and no
signs, can learn to sort out who goes first purely by watching each other move -- and show
that an asymmetric (leader/follower) reward is what makes that work, by running the exact
same learner with a symmetric reward and watching it deadlock.

## The claim being tested

Two claims, in order of importance.

1. **The negative claim.** A symmetric reward, learned with parameter-shared actors,
   collapses into mutual waiting. Both cars sit at the line. Nobody crosses.
2. **The positive claim.** Keeping absolutely everything else fixed and changing only the
   reward to a Stackelberg (leader/follower) one, the same learner instead learns to
   break the tie: one car commits, the other concedes, and both get through.

The comparison is the deliverable. A single trained policy that crosses an intersection is
not interesting -- plenty of things cross intersections. The interesting object is the
paired difference between two runs that differ in one line of config.

## v1 is done when all of these exist

- [ ] A completed Stackelberg training run: checkpoint, training log, final evaluation
      JSON, all under `runs/`.
- [ ] A completed symmetric training run at the same seed and the same environment-step
      budget, from the same commit.
- [ ] The two runs repeated over at least three seeds. Three is a choice, not a finding --
      it is the smallest number that shows whether a difference survives reseeding.
- [ ] `results/<tag>.json` written by `scripts/run_comparison.py`, holding both variants'
      per-seed and per-scenario numbers.
- [ ] A short results section in `README.md` quoting those numbers, with the exact command
      that produced them and the commit hash it ran at.
- [ ] A plot or table of deadlock rate and time-to-resolve for both variants across the
      eight fixed evaluation scenarios.
- [ ] An honest paragraph on what the result does *not* show.

## What counts as success

The criteria are directional on purpose. Absolute thresholds picked before any run would be
made up, and picking them afterwards is fitting the target to the result.

Success means, on the fixed evaluation suite:

- The Stackelberg variant's **deadlock rate is lower** than the symmetric variant's, by a
  margin bigger than the spread across seeds. The two head-on symmetric scenarios
  (`sym_slow`, `sym_fast`) are the ones that matter -- they are the cases where no
  geometric excuse to go first exists.
- The Stackelberg variant's **collision rate is not higher**. Trading deadlocks for crashes
  is not a result, it is a worse policy.
- **Time-to-resolve is lower** on the episodes that do resolve.
- **Leader switches per episode stay low.** If the leader label flips every few steps, the
  reward is chasing filter noise and any improvement is incidental.

## What counts as failure, and why that is still fine

Any of these ends v1 with a negative result written up honestly:

- Both variants deadlock. The asymmetry is not strong enough, or the shaping never fires
  because the two posteriors stay inside `leader_margin`. `leader_switches_mean` and the
  posterior traces say which.
- Both variants resolve. The symmetric baseline did not actually deadlock, so there was no
  problem to fix, and the whole premise needs rethinking.
- Stackelberg resolves but by crashing more often.
- The difference vanishes when the seed changes.

A written-up negative result is a finished v1. A missing run is not.

## Explicitly out of scope for v1

- More than two vehicles.
- Turning routes. Both cars go straight through; turns merge rather than cross and the
  right-of-way question mostly evaporates.
- Pedestrians, cyclists, background traffic.
- Any learned or human-modelled opponent that is not this same policy.
- Continuous throttle. The action space is three intents and that is the point.
- Vehicle-to-vehicle messaging. The whole project is about inferring intent from motion.
- **Any physical testbed.** There is no hardware, no scaled vehicles, no track. Real
  hardware is a possible later milestone and nothing more than that today.

## After v1

Rough order, none of it started:

1. Sensitivity: how much does the result depend on `leader_margin`, on the filter's
   particle count, on the assertiveness penalty?
2. A third reward variant so the comparison is not just two points.
3. Asymmetric opponents: freeze one agent's policy and train against it.
4. More arms, more vehicles, turning routes.
5. Only then, a scaled physical testbed.
