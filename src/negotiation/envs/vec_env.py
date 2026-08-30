from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..inference.particle_filter import FilterConfig
from ..metrics import EpisodeTracker
from .belief_wrapper import BeliefWrapper
from .intersection import NegotiationIntersectionEnv, Scenario
from .scenarios import sample_training_scenario


def make_env(scenario: Scenario | None = None,
             filter_config: FilterConfig | None = None,
             seed: int | None = None,
             config_overrides: dict | None = None) -> BeliefWrapper:
    config = dict(config_overrides or {})
    if scenario is not None:
        config["scenario"] = scenario.as_dict()
    env = NegotiationIntersectionEnv(config=config)
    return BeliefWrapper(env, filter_config=filter_config, seed=seed)


class SyncVecEnv:
    """A stack of environments stepped in one process.

    Everything is indexed by a *global* environment id (`env_offset + i`) rather than by
    position in this stack, and every environment carries its own RNG. That is what lets
    AsyncVecEnv split the same stack across worker processes and get bit-identical
    trajectories back -- a shared sampler RNG would make the results depend on how the
    envs happened to be partitioned.
    """

    def __init__(self, num_envs: int, seed: int = 0,
                 filter_config: FilterConfig | None = None,
                 scenario_sampler=sample_training_scenario,
                 fixed_scenarios: list[Scenario] | None = None,
                 config_overrides: dict | None = None,
                 reward_model=None,
                 env_offset: int = 0,
                 total_envs: int | None = None):
        self.num_envs = num_envs
        self.env_offset = env_offset
        self.total_envs = total_envs or num_envs
        self.seed = seed
        self.scenario_sampler = scenario_sampler
        self.fixed_scenarios = fixed_scenarios
        self.reward_model = reward_model

        gids = [env_offset + i for i in range(num_envs)]
        self.rngs = [np.random.default_rng([seed, g]) for g in gids]
        self.envs = [make_env(filter_config=filter_config, seed=seed + g,
                              config_overrides=config_overrides)
                     for g in gids]
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].state_dim
        self.dt = self.envs[0].dt
        wait_speed = self.envs[0].env.config["wait_speed"]
        self.trackers = [EpisodeTracker(self.dt, wait_speed) for _ in range(num_envs)]
        self._seeds = [seed * 1000 + g for g in gids]

    def spec(self) -> dict:
        return {"obs_dim": self.obs_dim, "state_dim": self.state_dim, "dt": self.dt}

    def _reset_one(self, i: int) -> np.ndarray:
        if self.fixed_scenarios is not None:
            scenario = self.fixed_scenarios[(self.env_offset + i) % len(self.fixed_scenarios)]
        else:
            scenario = self.scenario_sampler(self.rngs[i])
        self.envs[i].set_scenario(scenario)
        self._seeds[i] += self.total_envs
        obs = self.envs[i].reset(seed=self._seeds[i])
        self.trackers[i].reset(scenario.name)
        return obs

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack([self._reset_one(i) for i in range(self.num_envs)])
        return obs, states_from_obs(obs)

    def step(self, actions: np.ndarray, reward_model=None):
        """actions: (num_envs, 2). Returns per-agent rewards and any finished episodes."""
        reward_model = reward_model if reward_model is not None else self.reward_model
        if reward_model is None:
            raise ValueError("no reward model: pass one to step() or to the constructor")

        obs_out = np.zeros((self.num_envs, 2, self.obs_dim), dtype=np.float32)
        # Kept separately because a time-limit truncation still has to be bootstrapped
        # from the state the episode actually ended in, not from the reset state.
        final_states = np.zeros((self.num_envs, 2, self.state_dim), dtype=np.float32)
        rewards = np.zeros((self.num_envs, 2), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        finished = []

        for i, env in enumerate(self.envs):
            obs, terms, terminated, truncated, info = env.step(actions[i])
            posteriors = env.posterior_assertive()
            rewards[i] = reward_model(terms, posteriors)

            positions = np.stack([v.position for v in env.env.controlled_vehicles])
            self.trackers[i].update(terms, positions, reward_model.leader(posteriors))

            dones[i] = terminated
            truncateds[i] = truncated
            if terminated or truncated:
                final_states[i] = env.states(obs)
                finished.append(self.trackers[i].finish(info["outcome"]))
                obs_out[i] = self._reset_one(i)
            else:
                obs_out[i] = obs

        return (obs_out, states_from_obs(obs_out), rewards, dones, truncateds,
                final_states, finished)

    def close(self) -> None:
        pass


def states_from_obs(obs: np.ndarray) -> np.ndarray:
    """Centralised critic input for every env at once: self first, other second.

    obs is (envs, 2, obs_dim); obs[:, ::-1] is the same array with the two agents swapped,
    so one concatenate builds both rows for every environment.
    """
    return np.concatenate([obs, obs[:, ::-1]], axis=2)


def _worker(remote, parent_remote, kwargs) -> None:
    parent_remote.close()
    vec = SyncVecEnv(**kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(vec.step(data))
            elif cmd == "reset":
                remote.send(vec.reset())
            elif cmd == "spec":
                remote.send(vec.spec())
            else:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class AsyncVecEnv:
    """The same stack of environments, split across worker processes.

    The environment step is pure CPU work -- highway-env physics plus two 256-particle
    filters -- and it is what a run of this actually spends its time on, not the policy.
    Each worker owns a contiguous slice of the environments and steps all of them per
    message, so the IPC cost is one round trip per worker per step rather than one per
    environment, and it stops mattering as soon as a worker holds more than a handful.

    The reward model is evaluated inside the workers, which is why it is fixed at
    construction: it has to be shipped to them once, not per step.
    """

    def __init__(self, num_envs: int, num_workers: int, reward_model,
                 seed: int = 0, **kwargs):
        num_workers = max(1, min(num_workers, num_envs))
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.reward_model = reward_model
        self.closed = False

        bounds = np.linspace(0, num_envs, num_workers + 1).round().astype(int)
        self.slices = [(int(bounds[k]), int(bounds[k + 1])) for k in range(num_workers)]

        # spawn, not fork: the parent has usually already initialised CUDA by this point,
        # and forking a process with a live CUDA context is a well-known way to get a
        # hang that only shows up on someone else's machine.
        ctx = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for start, stop in self.slices:
            remote, child = ctx.Pipe()
            worker_kwargs = dict(kwargs)
            worker_kwargs.update(num_envs=stop - start, seed=seed, reward_model=reward_model,
                                 env_offset=start, total_envs=num_envs)
            p = ctx.Process(target=_worker, args=(child, remote, worker_kwargs), daemon=True)
            p.start()
            child.close()
            self.remotes.append(remote)
            self.processes.append(p)

        self.remotes[0].send(("spec", None))
        spec = self.remotes[0].recv()
        self.obs_dim, self.state_dim, self.dt = spec["obs_dim"], spec["state_dim"], spec["dt"]

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for remote in self.remotes:
            remote.send(("reset", None))
        parts = [remote.recv() for remote in self.remotes]
        obs = np.concatenate([p[0] for p in parts])
        return obs, np.concatenate([p[1] for p in parts])

    def step(self, actions: np.ndarray, reward_model=None):
        if reward_model is not None and reward_model is not self.reward_model:
            raise ValueError("AsyncVecEnv workers hold their own reward model; pass it "
                             "at construction instead of per step")
        for remote, (start, stop) in zip(self.remotes, self.slices):
            remote.send(("step", actions[start:stop]))
        parts = [remote.recv() for remote in self.remotes]

        finished = [e for p in parts for e in p[6]]
        return (*(np.concatenate([p[j] for p in parts]) for j in range(6)), finished)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.close()
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        self.close()


def auto_workers(num_envs: int) -> int:
    """Leave half the cores for the main process, which is doing the GPU feeding."""
    return max(1, min(num_envs, (os.cpu_count() or 2) // 2))


def make_vec_env(num_envs: int, reward_model=None, num_workers: int = 1, **kwargs):
    """num_workers: 1 for in-process, 0 for auto, n for n worker processes."""
    if num_workers == 0:
        num_workers = auto_workers(num_envs)
    if num_workers <= 1 or num_envs == 1:
        return SyncVecEnv(num_envs, reward_model=reward_model, **kwargs)
    if reward_model is None:
        raise ValueError("worker processes need the reward model at construction")
    return AsyncVecEnv(num_envs, num_workers, reward_model, **kwargs)
