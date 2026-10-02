"""Simple scripted agents sharing one interface: ``reset()`` and ``act(view, match)``."""

from __future__ import annotations

import math

import numpy as np

from .env import Match
from .state import View
from .teacher import TeacherBot


class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def act(self, view: View, match: Match) -> np.ndarray:
        raise NotImplementedError


class IdleAgent(Agent):
    name = "idle"

    def act(self, view, match):
        return np.zeros(8, np.float32)


class RandomAgent(Agent):
    """Holds random controls for ~0.3 s at a time."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros(8, np.float32)
        self.k = 0

    def act(self, view, match):
        if self.k <= 0:
            self.a = np.concatenate([self.rng.uniform(-1, 1, 5), self.rng.random(3) < 0.3]).astype(np.float32)
            self.a[0] = abs(self.a[0])
            self.k = 10
        self.k -= 1
        return self.a


class BallChaseAgent(Agent):
    """Drives straight at the ball with boost: the classic baseline bot."""
    name = "ballchase"

    def act(self, view, match):
        a = np.zeros(8, np.float32)
        loc = view.me.local(view.ball_pos - view.me.pos)
        ang = math.atan2(loc[1], loc[0])
        a[0] = 1.0
        a[1] = np.clip(3 * ang, -1, 1)
        a[6] = 1.0 if abs(ang) < 0.3 else 0.0
        a[7] = 1.0 if abs(ang) > 1.8 else 0.0
        return a


class TeacherAgent(Agent):
    name = "teacher"

    def __init__(self, seed: int = 0, aggression: float = 1.0):
        self.bot = TeacherBot(aggression=aggression, rng=np.random.default_rng(seed))

    def reset(self):
        self.bot.reset()

    def act(self, view, match):
        t, p = match.team_ball_prediction(view.team)
        return self.bot.act(view, t, p, match.big_pad_flags(view.team))


def play(blue: Agent, orange: Agent, seconds: float = 300.0, seed: int = 0,
         reset_mode: str = "kickoff") -> dict:
    """Play a match with kickoffs after every goal; returns score and touch counts."""
    m = Match(seed=seed, max_seconds=seconds, reset_mode=reset_mode)
    agents = [blue, orange]
    views = m.reset("kickoff")
    for ag in agents:
        ag.reset()
    total_steps = int(seconds * 120 / m.tick_skip)
    touches = [0, 0]
    for _ in range(total_steps):
        acts = [ag.act(v, m) for ag, v in zip(agents, views)]
        views, info = m.step(acts)
        touches[0] += info["touch"][0]
        touches[1] += info["touch"][1]
        if info["goal"] is not None:
            views = m.reset("kickoff")
            for ag in agents:
                ag.reset()
    return {"score": list(m.score), "touches": touches}
