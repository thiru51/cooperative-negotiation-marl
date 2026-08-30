from __future__ import annotations

from dataclasses import dataclass

from .envs.intents import Intent
from .envs.intersection import AgentTerms


@dataclass
class RewardConfig:
    progress: float = 0.30
    cleared: float = 5.0
    collision: float = 12.0
    wait: float = 0.12
    step: float = 0.02
    jerk: float = 0.004
    assertive: float = 0.25
    yield_bonus: float = 0.20
    follower_wait_discount: float = 0.20
    # How far apart the two "is that car assertive?" posteriors have to be before we are
    # willing to call one agent the leader. Too small and the roles thrash on filter noise
    # every step; too large and the shaping never fires at all.
    leader_margin: float = 0.15


class RewardModel:
    """Maps the environment's raw per-agent terms to scalars.

    Both variants share every task-relevant coefficient (progress, clearing the junction,
    collision, comfort). They differ only in whether the shaping on *signalling* is
    symmetric between the two agents. Evaluation metrics are computed from the raw terms,
    never from these scalars, so the two variants are scored on the same ruler.
    """

    variant = "base"

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()

    def _base(self, t: AgentTerms) -> float:
        c = self.cfg
        return (
            c.progress * t.progress
            + c.cleared * t.just_cleared
            - c.collision * t.collision
            - c.jerk * t.jerk
            - c.step
        )

    def __call__(self, terms, posteriors) -> list[float]:
        raise NotImplementedError

    def leader(self, posteriors) -> int | None:
        """posteriors[j] = P(agent j is assertive), as estimated by the *other* agent."""
        diff = posteriors[0] - posteriors[1]
        if abs(diff) < self.cfg.leader_margin:
            return None
        return 0 if diff > 0 else 1


class SymmetricReward(RewardModel):
    """Control condition: identical reward for both agents, including an equal penalty on
    signalling assertively. This is the reward that has a mutual-yield fixed point."""

    variant = "symmetric"

    def __call__(self, terms, posteriors) -> list[float]:
        c = self.cfg
        out = []
        for t in terms:
            r = self._base(t)
            r -= c.wait * t.waiting
            r -= c.assertive * float(t.intent == int(Intent.ASSERTIVE_ADVANCE))
            out.append(float(r))
        return out


class StackelbergReward(RewardModel):
    """Treatment condition: once the intention filters agree on who is behaving as leader,
    that agent stops paying the assertiveness cost and the other is paid to concede.

    The leader assignment is a training-time signal computed from both filters. At
    execution each agent still only sees its own filter's posterior -- this is the usual
    centralised-training / decentralised-execution split, applied to the reward rather
    than only to the critic.
    """

    variant = "stackelberg"

    def __call__(self, terms, posteriors) -> list[float]:
        c = self.cfg
        leader = self.leader(posteriors)
        engaged = not any(t.cleared for t in terms)

        out = []
        for i, t in enumerate(terms):
            r = self._base(t)
            assertive = float(t.intent == int(Intent.ASSERTIVE_ADVANCE))

            if leader is None or not engaged:
                r -= c.wait * t.waiting
                r -= c.assertive * assertive
            elif i == leader:
                r -= c.wait * t.waiting
            else:
                r -= c.wait * c.follower_wait_discount * t.waiting
                r -= c.assertive * assertive
                r += c.yield_bonus * float(t.intent == int(Intent.YIELD_NUDGE))
            out.append(float(r))
        return out


REWARD_MODELS = {
    "symmetric": SymmetricReward,
    "stackelberg": StackelbergReward,
}


def make_reward(variant: str, config: RewardConfig | None = None) -> RewardModel:
    if variant not in REWARD_MODELS:
        raise ValueError(f"unknown reward variant {variant!r}, expected one of {sorted(REWARD_MODELS)}")
    return REWARD_MODELS[variant](config)
