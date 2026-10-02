"""Batched fly-vs-opponent matches (one fly brain per batch column)."""

from __future__ import annotations

import numpy as np

from ..brain.controller import FlyController
from ..sim.agents import Agent
from ..sim.env import Match, step_reward
from ..sim.obs import build_obs


def fly_matches(ctrl: FlyController, matches: list[Match], opponents: list[Agent], n_steps: int,
                dopamine: bool = True, first_reset: str = "kickoff", after_goal: str = "kickoff") -> dict:
    """The fly drives car 0 (blue) in every match. Plastic weight changes persist across goals."""
    B = len(matches)
    views = [m.reset(first_reset) for m in matches]
    for o in opponents:
        o.reset()
    prev = np.zeros((B, 8), np.float32)
    reward = np.zeros(B, np.float32)
    out = {k: np.zeros(B) for k in ("fitness", "goals_for", "goals_against", "touches")}
    for _ in range(n_steps):
        obs = np.stack([build_obs(views[e][0], prev[e]) for e in range(B)])
        acts = ctrl.step(obs, reward if dopamine else None)
        reset_cols = []
        for e in range(B):
            opp_a = opponents[e].act(views[e][1], matches[e])
            new_views, info = matches[e].step([acts[e], opp_a])
            r = step_reward(new_views[0], views[e][0], info["touch"][0], info["goal"])
            reward[e] = r
            out["fitness"][e] += r
            out["touches"][e] += info["touch"][0]
            prev[e] = acts[e]
            views[e] = new_views
            if info["goal"] is not None:
                out["goals_for" if info["goal"] == 0 else "goals_against"][e] += 1
                views[e] = matches[e].reset(after_goal)
                opponents[e].reset()
                prev[e] = 0
                reset_cols.append(e)
        if reset_cols:
            ctrl.reset_columns(reset_cols)
    return out
