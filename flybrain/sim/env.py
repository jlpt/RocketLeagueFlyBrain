"""RocketSim 1v1 match wrapper used for training, evaluation and play."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import RocketSim as rs

from . import arena_mesh
from .state import OPP_GOAL, View, views_from_arena

ACTION_DIM = 8          # throttle, steer, pitch, yaw, roll, jump, boost, handbrake
N_ANALOG = 5
PHYSICS_HZ = 120

_INIT_LOCK = threading.Lock()
_INITIALISED = False


def ensure_rocketsim(meshes: str | os.PathLike | None = None) -> str:
    """Initialise RocketSim once. Uses real dumped meshes if available, else the generated arena."""
    global _INITIALISED
    with _INIT_LOCK:
        if _INITIALISED:
            return ""
        path = meshes or os.environ.get("RS_COLLISION_MESHES")
        if path is None or not (Path(path) / "soccar").exists():
            path = os.environ.get("FLYBRAIN_GENERATED_MESHES", "collision_meshes")
            if not (Path(path) / "soccar").exists():
                arena_mesh.generate(path)
        rs.init(str(path))
        _INITIALISED = True
        return str(path)


def to_controls(a: np.ndarray) -> rs.CarControls:
    c = rs.CarControls()
    c.throttle = float(np.clip(a[0], -1, 1))
    c.steer = float(np.clip(a[1], -1, 1))
    c.pitch = float(np.clip(a[2], -1, 1))
    c.yaw = float(np.clip(a[3], -1, 1))
    c.roll = float(np.clip(a[4], -1, 1))
    c.jump = bool(a[5] > 0.5)
    c.boost = bool(a[6] > 0.5)
    c.handbrake = bool(a[7] > 0.5)
    return c


class Match:
    """A 1v1 soccar match: car 0 is blue, car 1 is orange.

    ``step`` advances ``tick_skip`` physics ticks (one brain step).
    """

    def __init__(self, seed: int = 0, tick_skip: int = 4, max_seconds: float = 60.0,
                 reset_mode: str = "mixed", kickoff_prob: float = 0.3, meshes=None):
        ensure_rocketsim(meshes)
        self.rng = np.random.default_rng(seed)
        self.tick_skip = tick_skip
        self.max_steps = int(max_seconds * PHYSICS_HZ / tick_skip)
        self.reset_mode = reset_mode
        self.kickoff_prob = kickoff_prob
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = [self.arena.add_car(rs.Team.BLUE), self.arena.add_car(rs.Team.ORANGE)]
        self._goal: int | None = None
        self._touch = [False, False]
        self.arena.set_goal_score_callback(self._on_goal)
        self.arena.set_ball_touch_callback(self._on_touch)
        self.steps = 0
        self.score = [0, 0]
        self._pred = None
        self._pred_step = -1
        # big boost pads, matched to teacher.BIG_PADS order for each team frame
        from .teacher import BIG_PADS
        pads = self.arena.get_boost_pads()
        self._pads = [p for p in pads if p.is_big]
        world = np.array([p.get_pos().as_numpy() for p in self._pads])
        self._pad_order = []
        for team in (0, 1):
            w = world.copy()
            if team == 1:
                w[:, :2] *= -1
            self._pad_order.append([int(np.argmin(np.linalg.norm(w[:, :2] - bp[:2], axis=1))) for bp in BIG_PADS])

    def ball_prediction(self, horizon_s: float = 2.5, every_ticks: int = 4):
        """Predicted ball positions (world frame): times (K,), positions (K,3)."""
        n = int(horizon_s * PHYSICS_HZ / every_ticks)
        pred = self.arena.get_ball_prediction(n, every_ticks)
        pos = np.array([s.pos.as_numpy() for s in pred], dtype=np.float64)
        t = (np.arange(1, n + 1) * every_ticks) / PHYSICS_HZ
        return t, pos

    def team_ball_prediction(self, team: int):
        """Ball prediction in a team's frame, cached per brain step."""
        if self._pred is None or self._pred_step != self.steps:
            self._pred = self.ball_prediction()
            self._pred_step = self.steps
        t, p = self._pred
        if team == 1:
            p = p * np.array([-1.0, -1.0, 1.0])
        return t, p

    def big_pad_flags(self, team: int) -> np.ndarray:
        active = np.array([float(p.get_state().is_active) for p in self._pads])
        return active[self._pad_order[team]]

    # ----------------------------------------------------------- callbacks
    def _on_goal(self, arena=None, team=None, data=None, **kw):
        if self._goal is None:
            self._goal = int(team if team is not None else kw.get("scoring_team"))

    def _on_touch(self, arena=None, car=None, data=None, **kw):
        car = car if car is not None else kw.get("car")
        if car is not None:
            self._touch[0 if int(car.team) == 0 else 1] = True

    # ---------------------------------------------------------------- reset
    def reset(self, mode: str | None = None) -> list[View]:
        mode = mode or self.reset_mode
        if mode == "mixed":
            mode = "kickoff" if self.rng.random() < self.kickoff_prob else "random"
        self.arena.reset_kickoff(int(self.rng.integers(0, 2**31 - 1)))
        if mode == "random":
            self._random_state()
        self._goal = None
        self._touch = [False, False]
        self.steps = 0
        self._pred = None
        return self.views()

    def _random_state(self) -> None:
        r = self.rng
        b = self.arena.ball.get_state()
        in_air = r.random() < 0.3
        b.pos = rs.Vec(r.uniform(-3300, 3300), r.uniform(-4000, 4000), r.uniform(150, 1200) if in_air else 93.15)
        ang = r.uniform(-np.pi, np.pi)
        sp = r.uniform(0, 1800)
        b.vel = rs.Vec(sp * np.cos(ang), sp * np.sin(ang), r.uniform(-200, 800) if in_air else 0.0)
        b.ang_vel = rs.Vec(0, 0, 0)
        self.arena.ball.set_state(b)
        for car in self.cars:
            s = car.get_state()
            yaw = r.uniform(-np.pi, np.pi)
            s.pos = rs.Vec(r.uniform(-3500, 3500), r.uniform(-4400, 4400), 17.0)
            s.rot_mat = rs.Angle(yaw, 0.0, 0.0).as_rot_mat()
            sp = r.uniform(0, 1700)
            s.vel = rs.Vec(sp * np.cos(yaw), sp * np.sin(yaw), 0.0)
            s.ang_vel = rs.Vec(0, 0, 0)
            s.boost = float(r.uniform(0, 100))
            car.set_state(s)

    # ----------------------------------------------------------------- step
    def views(self) -> list[View]:
        return views_from_arena(self.arena, self.cars, self.steps * self.tick_skip / PHYSICS_HZ)

    def step(self, actions: list[np.ndarray]) -> tuple[list[View], dict]:
        for car, a in zip(self.cars, actions):
            car.set_controls(to_controls(a))
        self._touch = [False, False]
        self.arena.step(self.tick_skip)
        self.steps += 1
        goal = self._goal
        if goal is not None:
            self.score[goal] += 1
        info = {"goal": goal, "touch": list(self._touch),
                "timeout": self.steps >= self.max_steps}
        info["done"] = goal is not None or info["timeout"]
        return self.views(), info


def step_reward(view: View, prev_view: View, touched: bool, goal: int | None) -> float:
    """Dense shaping + sparse outcome reward from one car's perspective (team frame)."""
    r = 0.0
    if goal is not None:
        r += 10.0 if goal == view.team else -10.0
    if touched:
        r += 0.3
    to_goal = OPP_GOAL - view.ball_pos
    to_goal /= np.linalg.norm(to_goal) + 1e-6
    r += 0.05 * float(view.ball_vel @ to_goal) / 2300.0
    to_ball = view.ball_pos - view.me.pos
    to_ball /= np.linalg.norm(to_ball) + 1e-6
    r += 0.01 * float(view.me.vel @ to_ball) / 2300.0
    return r
