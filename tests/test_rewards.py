from __future__ import annotations

import pytest

from negotiation.envs.intents import Intent
from negotiation.envs.intersection import AgentTerms
from negotiation.rewards import RewardConfig, StackelbergReward, SymmetricReward, make_reward


def terms(intent=Intent.CREEP, waiting=0.0, **kw) -> AgentTerms:
    return AgentTerms(intent=int(intent), waiting=waiting, **kw)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        make_reward("nash")


def test_leader_is_undecided_inside_the_margin():
    r = make_reward("stackelberg", RewardConfig(leader_margin=0.15))
    assert r.leader([0.5, 0.5]) is None
    assert r.leader([0.6, 0.5]) is None
    assert r.leader([0.8, 0.2]) == 0
    assert r.leader([0.2, 0.8]) == 1


def test_symmetric_reward_is_permutation_invariant():
    """The control condition must be exactly symmetric, otherwise the comparison is
    confounded by an accidental tie-break hidden in the baseline."""
    r = SymmetricReward()
    a = terms(Intent.ASSERTIVE_ADVANCE, progress=1.2, speed=6.0)
    b = terms(Intent.YIELD_NUDGE, waiting=1.0, speed=0.0)
    forward = r([a, b], [0.9, 0.1])
    backward = r([b, a], [0.1, 0.9])
    assert forward == list(reversed(backward))


def test_symmetric_reward_ignores_the_posterior_entirely():
    r = SymmetricReward()
    pair = [terms(Intent.ASSERTIVE_ADVANCE, progress=1.0), terms(Intent.YIELD_NUDGE, waiting=1.0)]
    assert r(pair, [0.5, 0.5]) == r(pair, [0.95, 0.05])


def test_symmetric_reward_penalises_both_agents_for_advancing():
    c = RewardConfig()
    r = SymmetricReward(c)
    quiet = r([terms(Intent.CREEP), terms(Intent.CREEP)], [0.5, 0.5])
    pushy = r([terms(Intent.ASSERTIVE_ADVANCE), terms(Intent.ASSERTIVE_ADVANCE)], [0.5, 0.5])
    assert pushy[0] == pytest.approx(quiet[0] - c.assertive)
    assert pushy[1] == pytest.approx(quiet[1] - c.assertive)


def test_stackelberg_matches_symmetric_when_no_leader_is_identified():
    c = RewardConfig()
    pair = [terms(Intent.ASSERTIVE_ADVANCE, progress=0.8), terms(Intent.YIELD_NUDGE, waiting=1.0)]
    assert StackelbergReward(c)(pair, [0.5, 0.5]) == pytest.approx(SymmetricReward(c)(pair, [0.5, 0.5]))


def test_leader_stops_paying_the_assertiveness_cost():
    c = RewardConfig()
    r = StackelbergReward(c)
    pair = [terms(Intent.ASSERTIVE_ADVANCE, progress=1.0), terms(Intent.YIELD_NUDGE, waiting=1.0)]
    undecided = r(pair, [0.5, 0.5])
    led = r(pair, [0.95, 0.05])
    assert led[0] == pytest.approx(undecided[0] + c.assertive)


def test_follower_is_paid_to_concede_and_charged_less_for_waiting():
    c = RewardConfig()
    r = StackelbergReward(c)
    pair = [terms(Intent.ASSERTIVE_ADVANCE, progress=1.0), terms(Intent.YIELD_NUDGE, waiting=1.0)]
    undecided = r(pair, [0.5, 0.5])
    led = r(pair, [0.95, 0.05])
    expected = undecided[1] + c.yield_bonus + c.wait * (1.0 - c.follower_wait_discount)
    assert led[1] == pytest.approx(expected)


def test_shaping_switches_off_once_someone_is_through():
    """Shaping is about resolving a live conflict. Once a car has cleared, leaving the
    bonus on would pay the other agent to keep signalling at an empty junction."""
    c = RewardConfig()
    r = StackelbergReward(c)
    pair = [terms(Intent.ASSERTIVE_ADVANCE, cleared=1.0), terms(Intent.YIELD_NUDGE, waiting=1.0)]
    assert r(pair, [0.95, 0.05]) == pytest.approx(SymmetricReward(c)(pair, [0.95, 0.05]))


def test_collision_dominates_every_other_term():
    c = RewardConfig()
    for r in (SymmetricReward(c), StackelbergReward(c)):
        crashed = r([terms(Intent.ASSERTIVE_ADVANCE, progress=2.0, collision=1.0),
                     terms(Intent.CREEP, collision=1.0)], [0.95, 0.05])
        assert all(v < -5.0 for v in crashed), (r.variant, crashed)


def test_both_variants_agree_on_the_task_terms():
    """Progress, clearing and collision must be identical across the two variants; only
    the signalling shaping is allowed to differ."""
    c = RewardConfig()
    sym, stack = SymmetricReward(c), StackelbergReward(c)
    pair = [terms(Intent.CREEP, progress=1.5, just_cleared=1.0), terms(Intent.CREEP, progress=0.4)]
    assert sym(pair, [0.5, 0.5]) == pytest.approx(stack(pair, [0.5, 0.5]))
