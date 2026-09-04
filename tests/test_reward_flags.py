"""Every reward coefficient the CLI exposes must actually reach the reward model.

run_comparison.py builds its config from flags and never reads configs/*.yaml, so a
coefficient that is not wired through here is silently unchangeable -- a sweep over it
would re-run the identical experiment and look like a null result. That has happened
once already, with entropy_coef.
"""
import argparse

import pytest

from negotiation.cli import REWARD_FIELDS, add_runtime_flags, apply_flags
from negotiation.rewards import RewardConfig, make_reward
from negotiation.training import TrainConfig


def parse(argv):
    p = argparse.ArgumentParser()
    add_runtime_flags(p)
    return p.parse_args(argv)


def test_collision_penalty_reaches_the_config():
    cfg = apply_flags(TrainConfig(), parse(["--collision-penalty", "48"]))
    assert cfg.reward.collision == 48.0


def test_collision_penalty_reaches_the_reward_model():
    cfg = apply_flags(TrainConfig(variant="stackelberg"), parse(["--collision-penalty", "48"]))
    model = make_reward(cfg.variant, cfg.reward)
    assert model.cfg.collision == 48.0


@pytest.mark.parametrize("field", REWARD_FIELDS)
def test_every_declared_reward_field_has_a_working_flag(field):
    """Guards the tuple itself: adding a name without a flag would pass silently."""
    flag = {"collision": "--collision-penalty", "cleared": "--cleared-bonus",
            "yield_bonus": "--yield-bonus", "assertive": "--assertive-coef"}[field]
    cfg = apply_flags(TrainConfig(), parse([flag, "3.5"]))
    assert getattr(cfg.reward, field) == 3.5


def test_omitting_a_flag_leaves_the_default_alone():
    cfg = apply_flags(TrainConfig(), parse([]))
    assert cfg.reward.collision == RewardConfig().collision == 12.0


def test_reward_flags_do_not_leak_into_mappo():
    cfg = apply_flags(TrainConfig(), parse(["--collision-penalty", "48"]))
    assert not hasattr(cfg.mappo, "collision")
