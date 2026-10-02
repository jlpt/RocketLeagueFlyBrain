"""Convert an RLBot ``GameTickPacket`` into the fly's team-frame ``View``.

Only attribute access is used, so this works with RLBot v4 packets and with
any object exposing the same fields (the tests use RocketSim-backed fakes).
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.state import CarView, View

_FLIP = np.array([-1.0, -1.0, 1.0])


def euler_to_rot(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Rocket League Euler angles -> rows (forward, right, up), RocketSim convention."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr])
    up = np.array([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr])
    return np.stack([fwd, right, up])


def _vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


class CarTracker:
    """Reconstructs flip availability, which RLBot does not report directly."""

    def __init__(self):
        self.jump_time = None
        self.dodge_time = None
        self.was_jumped = False
        self.was_double = False

    def update(self, car, t: float) -> tuple[bool, bool, bool]:
        on_ground = bool(car.has_wheel_contact)
        if on_ground:
            self.jump_time = None
            self.dodge_time = None
        if car.jumped and not self.was_jumped:
            self.jump_time = t
        if car.double_jumped and not self.was_double:
            self.dodge_time = t
        self.was_jumped, self.was_double = bool(car.jumped), bool(car.double_jumped)
        if on_ground:
            has_flip = True
        elif car.double_jumped:
            has_flip = False
        else:
            has_flip = self.jump_time is None or (t - self.jump_time) < 1.25
        is_flipping = (not on_ground and self.dodge_time is not None and (t - self.dodge_time) < 0.65)
        return has_flip, bool(car.jumped), is_flipping


def _car_view(car, tracker: CarTracker, t: float, flip: bool) -> CarView:
    ph = car.physics
    rot = euler_to_rot(ph.rotation.pitch, ph.rotation.yaw, ph.rotation.roll)
    pos, vel, ang = _vec(ph.location), _vec(ph.velocity), _vec(ph.angular_velocity)
    if flip:
        pos, vel, ang = pos * _FLIP, vel * _FLIP, ang * _FLIP
        rot = rot * _FLIP[None, :]
    has_flip, has_jumped, is_flipping = tracker.update(car, t)
    return CarView(pos=pos, vel=vel, ang_vel=ang, fwd=rot[0], right=rot[1], up=rot[2],
                   boost=float(car.boost), on_ground=bool(car.has_wheel_contact), has_flip=has_flip,
                   has_jumped=has_jumped, is_flipping=is_flipping, demoed=bool(car.is_demolished))


def view_from_packet(packet, index: int, trackers: dict) -> View:
    me_car = packet.game_cars[index]
    team = int(me_car.team)
    flip = team == 1
    t = float(packet.game_info.seconds_elapsed)
    for i in range(packet.num_cars):
        trackers.setdefault(i, CarTracker())
    me = _car_view(me_car, trackers[index], t, flip)
    opp = None
    for i in range(packet.num_cars):
        c = packet.game_cars[i]
        if i != index and int(c.team) != team:
            opp = _car_view(c, trackers[i], t, flip)
            break
    b = packet.game_ball.physics
    bp, bv, ba = _vec(b.location), _vec(b.velocity), _vec(b.angular_velocity)
    if flip:
        bp, bv, ba = bp * _FLIP, bv * _FLIP, ba * _FLIP
    return View(me=me, opp=opp, ball_pos=bp, ball_vel=bv, ball_ang_vel=ba, team=team, time=t)
