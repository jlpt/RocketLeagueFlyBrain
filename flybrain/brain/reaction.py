"""Measure the fly's visual reaction time, like a human simple-reaction-time test.

Protocol: hold a game situation until the brain settles, then change only the
*visual* channel (the ball, opponent and goals jump to where they are in a
different situation) and watch the car controls. Reaction time = time from the
visual change until the controls have moved ``threshold`` of the way to their
new steady state. Reference: human simple visual RT ~200-250 ms, Rocket
League players ~150-250 ms.
"""

from __future__ import annotations

import numpy as np
import torch

from ..sim.env import Match
from ..sim.obs import GROUP_SLICES, build_obs
from .controller import FlyController
from .model import N_ANALOG, FlyBrain


def random_obs(n: int, seed: int = 0) -> np.ndarray:
    m = Match(seed=seed)
    out = []
    for _ in range(n):
        v = m.reset("random")[0]
        out.append(build_obs(v, np.zeros(8, np.float32)))
    return np.array(out, dtype=np.float32)


@torch.no_grad()
def measure_reaction_time(brain: FlyBrain, n_pairs: int = 64, settle: int = 45, horizon: int = 45,
                          threshold: float = 0.5, seed: int = 0) -> dict:
    a_obs = random_obs(n_pairs, seed)
    b_src = random_obs(n_pairs, seed + 1)
    b_obs = a_obs.copy()
    vis = GROUP_SLICES["in_visual"]
    b_obs[:, vis] = b_src[:, vis]

    ctrl = FlyController(brain, n_pairs)
    acts = []
    for _ in range(settle):
        a = ctrl.step(a_obs)
    a0 = a[:, :N_ANALOG].copy()
    for _ in range(horizon):
        acts.append(ctrl.step(b_obs)[:, :N_ANALOG].copy())
    acts = np.stack(acts)                       # (horizon, n, 5)
    a_final = acts[-5:].mean(0)
    delta = np.linalg.norm(a_final - a0, axis=1)
    resp = np.linalg.norm(acts - a0[None], axis=2)  # (horizon, n)
    step_ms = brain.cfg.step_ms
    rts, onsets = [], []
    for i in range(n_pairs):
        if delta[i] < 0.1:
            continue
        k = np.where(resp[:, i] >= threshold * delta[i])[0]
        k0 = np.where(resp[:, i] >= 0.1 * delta[i])[0]
        if len(k):
            rts.append(k[0] * step_ms)
        if len(k0):
            onsets.append(k0[0] * step_ms)
    rts, onsets = np.array(rts), np.array(onsets)
    c = brain.cfg
    floor = (c.sensory_delay + 1 + c.motor_delay) * step_ms
    return {
        "n_responding": int(len(rts)),
        "rt_median_ms": float(np.median(rts)) if len(rts) else float("nan"),
        "rt_p25_ms": float(np.percentile(rts, 25)) if len(rts) else float("nan"),
        "rt_p75_ms": float(np.percentile(rts, 75)) if len(rts) else float("nan"),
        "onset_median_ms": float(np.median(onsets)) if len(onsets) else float("nan"),
        "hard_floor_ms": floor,
        "threshold": threshold,
    }
