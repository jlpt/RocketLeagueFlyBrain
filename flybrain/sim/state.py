"""Team-perspective views of a RocketSim arena.

Everything the bots (fly or teacher) see goes through ``View``: positions and
directions are expressed so that every player attacks towards +y. For the
orange team this is RocketSim's "inverse" gym state (rotated 180 deg about z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# get_gym_state() car layout
C_ON_GROUND, C_BOOST = 8, 10
C_POS, C_VEL, C_ANGVEL = slice(11, 14), slice(18, 21), slice(21, 24)
C_FWD, C_RIGHT, C_UP = slice(24, 27), slice(27, 30), slice(30, 33)
C_DEMOED = 7
# ball layout
B_POS, B_VEL, B_ANGVEL = slice(0, 3), slice(7, 10), slice(10, 13)

ARENA_X, ARENA_Y, ARENA_Z = 4096.0, 5120.0, 2044.0
GOAL_Y = 5120.0
BALL_RADIUS = 91.25
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANGVEL = 5.5
OPP_GOAL = np.array([0.0, GOAL_Y, 321.0])
OWN_GOAL = np.array([0.0, -GOAL_Y, 321.0])


@dataclass
class CarView:
    pos: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray
    fwd: np.ndarray
    right: np.ndarray
    up: np.ndarray
    boost: float           # 0..100
    on_ground: bool
    has_flip: bool         # can still jump / flip
    has_jumped: bool
    is_flipping: bool
    demoed: bool

    @property
    def rot(self) -> np.ndarray:
        """World->local rotation (rows: forward, right, up)."""
        return np.stack([self.fwd, self.right, self.up])

    def local(self, v: np.ndarray) -> np.ndarray:
        return self.rot @ v


@dataclass
class View:
    me: CarView
    opp: CarView | None
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_ang_vel: np.ndarray
    team: int              # 0 blue, 1 orange
    time: float = 0.0      # seconds since episode start


def _car_view(arr: np.ndarray, cstate) -> CarView:
    return CarView(
        pos=arr[C_POS].astype(np.float64), vel=arr[C_VEL].astype(np.float64),
        ang_vel=arr[C_ANGVEL].astype(np.float64), fwd=arr[C_FWD].astype(np.float64),
        right=arr[C_RIGHT].astype(np.float64), up=arr[C_UP].astype(np.float64),
        boost=float(arr[C_BOOST]), on_ground=bool(arr[C_ON_GROUND] > 0.5),
        has_flip=bool(cstate.has_flip_or_jump), has_jumped=bool(cstate.has_jumped),
        is_flipping=bool(cstate.is_flipping), demoed=bool(arr[C_DEMOED] > 0.5),
    )


def views_from_arena(arena, cars, t: float = 0.0) -> list[View]:
    """One View per car in ``cars`` (order of ``arena.get_cars()``)."""
    gs = arena.get_gym_state()
    ball = gs[2]
    car_arrays = gs[3:]
    states = [c.get_state() for c in cars]
    out = []
    for i, car in enumerate(cars):
        team = int(car.team)
        p = 1 if team == 1 else 0          # orange sees the rotated world
        me = _car_view(car_arrays[i][p], states[i])
        opp = None
        for j in range(len(cars)):
            if j != i and int(cars[j].team) != team:
                opp = _car_view(car_arrays[j][p], states[j])
                break
        b = ball[p]
        out.append(View(me=me, opp=opp, ball_pos=b[B_POS].astype(np.float64),
                        ball_vel=b[B_VEL].astype(np.float64),
                        ball_ang_vel=b[B_ANGVEL].astype(np.float64), team=team, time=t))
    return out


def flip_vec(v: np.ndarray) -> np.ndarray:
    """Convert a team-perspective vector for the orange team back to world (and vice versa)."""
    return np.array([-v[0], -v[1], v[2]])
