from __future__ import annotations

import torch

from negotiation.device import peak_memory, setup_device
from negotiation.envs.intents import N_INTENTS
from negotiation.rl.mappo import MAPPO, MAPPOConfig


def test_cpu_setup_has_no_amp():
    dev = setup_device("cpu")
    assert dev.device.type == "cpu"
    assert dev.amp_dtype is None
    assert not dev.needs_scaler
    assert peak_memory(dev)["peak_allocated_gb"] is None


def test_scaler_is_only_enabled_for_fp16():
    """bf16 keeps float32's exponent range, so a loss scaler is unnecessary there and
    switching it on would rescale gradients before clipping for no reason."""
    dev = setup_device("cpu")
    dev.amp_dtype = torch.bfloat16
    assert not dev.needs_scaler
    assert not dev.scaler().is_enabled()

    dev.amp_dtype = torch.float16
    assert dev.needs_scaler


def test_auto_never_hardcodes_cuda():
    dev = setup_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.device.type == expected


def test_batch_size_overrides_the_minibatch_count():
    cfg = MAPPOConfig(horizon=8, num_envs=4, hidden=16, batch_size=16)
    agent = MAPPO(6, 12, N_INTENTS, cfg, seed=0)
    buffer = agent.make_buffer()
    assert buffer.num_samples == 8 * 4 * 2
    assert agent.num_minibatches(buffer) == 4

    agent.cfg.batch_size = None
    agent.cfg.num_minibatches = 2
    assert agent.num_minibatches(buffer) == 2
