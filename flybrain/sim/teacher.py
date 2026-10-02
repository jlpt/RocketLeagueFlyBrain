"""A hand-written 1v1 bot used as the imitation-learning teacher (and as a sparring partner).

It is a classic "intercept + aim" bot: it reads RocketSim's ball prediction,
picks the earliest ball position it can reach in time, approaches from behind
the ball so the hit sends it towards the opponent's net (or away from its own
net when defending), dodges into the ball, recovers in the air, grabs boost
when low and rotates back post when it is caught on the wrong side.

All reasoning is done in the team frame (own goal at -y, opponent goal at +y).
"""

from __future__ import annotations

import math

import numpy as np

from .state import BALL_RADIUS, GOAL_Y, View

BIG_PADS = np.array([[-3584, 0, 73], [3584, 0, 73], [-3072, 4096, 73], [3072, 4096, 73],
                     [-3072, -4096, 73], [3072, -4096, 73]], dtype=np.float64)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


class TeacherBot:
    def __init__(self, aggression: float = 1.0, rng: np.random.Generator | None = None):
        self.aggression = aggression
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.flip_step = -1          # >= 0 while executing a dodge
        self.flip_dir = (0.0, 0.0)
        self.stuck_steps = 0
        self.reverse_steps = 0
        self.reverse_steer = 0.0

    # ----------------------------------------------------------- primitives
    @staticmethod
    def _angle_to(v: View, target: np.ndarray) -> float:
        loc = v.me.local(target - v.me.pos)
        return math.atan2(loc[1], loc[0])

    def _drive(self, v: View, target: np.ndarray, speed: float, a: np.ndarray, allow_boost=True) -> float:
        me = v.me
        ang = self._angle_to(v, target)
        yaw_rate = float(me.ang_vel @ me.up)
        a[1] = np.clip(3.0 * ang - 0.15 * yaw_rate, -1, 1)
        fwd_speed = float(me.vel @ me.fwd)
        if abs(ang) > 2.2 and np.linalg.norm(target - me.pos) < 600 and fwd_speed < 600:
            # target right behind us and close: reverse towards it
            a[0] = -1.0
            a[1] = -np.sign(ang) * 1.0 if abs(ang) < 2.9 else 0.0
            return ang
        if speed > fwd_speed + 40:
            a[0] = 1.0
            if (allow_boost and speed > 1400 and fwd_speed < speed - 100 and abs(ang) < 0.35
                    and me.boost > 0 and me.on_ground):
                a[6] = 1.0
        elif speed < fwd_speed - 250:
            a[0] = -1.0 if fwd_speed > 0 else 0.0
        else:
            a[0] = 0.05
        if abs(ang) > 1.7 and fwd_speed > 700 and me.on_ground and me.up[2] > 0.8:
            a[7] = 1.0
        return ang

    def _recover(self, v: View, a: np.ndarray, face: np.ndarray) -> None:
        me = v.me
        pitch_rate_up = -float(me.ang_vel @ me.right)
        roll_rate_right = -float(me.ang_vel @ me.fwd)
        yaw_rate = float(me.ang_vel @ me.up)
        a[2] = np.clip(-3.0 * me.fwd[2] - 0.5 * pitch_rate_up, -1, 1)
        if me.up[2] < 0.0:
            a[4] = 1.0 if me.right[2] >= 0 else -1.0
        else:
            a[4] = np.clip(3.0 * me.right[2] - 0.5 * roll_rate_right, -1, 1)
        loc = me.local(face)
        a[3] = np.clip(2.0 * math.atan2(loc[1], loc[0]) - 0.4 * yaw_rate, -1, 1)
        a[0] = 1.0  # throttle so we keep momentum on landing

    def _start_flip(self, v: View, target: np.ndarray) -> None:
        loc = v.me.local(target - v.me.pos)
        ang = math.atan2(loc[1], loc[0])
        self.flip_dir = (-math.cos(ang), math.sin(ang))   # (pitch, yaw)
        self.flip_step = 0

    def _continue_flip(self, v: View, a: np.ndarray) -> bool:
        s = self.flip_step
        if s < 0:
            return False
        # The teacher also labels states the student drove into (DAgger), so the dodge
        # sequence must follow what actually happened: abort if we never left the ground.
        if s >= 2 and v.me.on_ground:
            self.flip_step = -1
            return False
        a[0] = 1.0
        if s == 0:
            a[5] = 1.0              # first jump (held one brain step = 33 ms)
        elif s == 1:
            a[5] = 0.0              # release
        elif s == 2:
            a[5] = 1.0              # dodge
            a[2], a[3] = self.flip_dir
        elif s < 18:
            a[2], a[3] = self.flip_dir   # hold direction through the flip
            if s > 10 and not v.me.is_flipping:
                self.flip_step = -1
                return False
        else:
            self.flip_step = -1
            return False
        self.flip_step += 1
        return True

    # ---------------------------------------------------------------- brain
    def _shot_dir(self, v: View, p: np.ndarray) -> np.ndarray:
        if p[1] < -2500 and abs(p[0]) < 2500:
            # defensive clear: away from our net, towards the side of the field
            d = p - np.array([0.0, -GOAL_Y - 600, 0.0])
            d[2] = 0
            return _norm(d)
        aim = np.array([np.clip(p[0], -600, 600) * 0.6, GOAL_Y + 300, 0.0])
        d = aim - p
        d[2] = 0
        return _norm(d)

    def _pick_intercept(self, v: View, pred_t: np.ndarray, pred_p: np.ndarray):
        me = v.me
        speed = float(np.linalg.norm(me.vel))
        vmax = 2300.0 if me.boost > 10 else 1410.0
        for k in range(len(pred_t)):
            p = pred_p[k]
            if p[2] > 260:
                continue
            sd = self._shot_dir(v, p)
            target = p - sd * (BALL_RADIUS + 50)
            d = float(np.linalg.norm((target - me.pos)[:2]))
            ang = abs(self._angle_to(v, target))
            v_avg = max(500.0, 0.5 * (max(speed, 0.0) + vmax)) if d > 800 else max(500.0, speed)
            eta = 0.35 * ang + d / v_avg
            if eta <= pred_t[k]:
                return k, sd
        # nothing reachable on the ground: go to wherever the ball comes down last in the horizon
        low = np.where(pred_p[:, 2] <= 260)[0]
        k = int(low[-1]) if len(low) else len(pred_t) - 1
        return k, self._shot_dir(v, pred_p[k])

    def act(self, v: View, pred_t: np.ndarray, pred_p: np.ndarray,
            pads: np.ndarray | None = None) -> np.ndarray:
        """pred_t: (K,) seconds ahead; pred_p: (K,3) team-frame ball positions;
        pads: optional (6,) activity flags of the big boost pads (team frame order of BIG_PADS)."""
        me = v.me
        if self.reverse_steps > 0:            # backing out after getting wedged on a post / wall
            self.reverse_steps -= 1
            a = np.zeros(8, dtype=np.float32)
            a[0], a[1] = -1.0, self.reverse_steer
            return a
        a = self._act(v, pred_t, pred_p, pads)
        if me.on_ground and self.flip_step < 0 and a[0] > 0.5 and np.linalg.norm(me.vel) < 120:
            self.stuck_steps += 1
            if self.stuck_steps > 20:          # ~0.7 s pushing without moving
                self.stuck_steps = 0
                self.reverse_steps = 20
                self.reverse_steer = -float(np.sign(a[1]) or 1.0)
        else:
            self.stuck_steps = 0
        return a

    def _act(self, v: View, pred_t: np.ndarray, pred_p: np.ndarray, pads: np.ndarray | None) -> np.ndarray:
        a = np.zeros(8, dtype=np.float32)
        me = v.me
        if me.demoed:
            return a
        if self._continue_flip(v, a):
            return a

        ball = v.ball_pos
        # ---------------------------------------------------------- airborne
        if not me.on_ground:
            vel_h = me.vel.copy()
            vel_h[2] = 0
            face = me.pos + (vel_h if np.linalg.norm(vel_h) > 300 else (ball - me.pos))
            self._recover(v, a, face)
            # opportunistic touch: ball right in front while in the air after a jump
            if me.has_flip and np.linalg.norm(ball - me.pos) < 220 and me.has_jumped:
                self._start_flip(v, ball)
                self.flip_step = 2
                self._continue_flip(v, a)
            return a

        # ---------------------------------------------------------- kickoff
        if np.linalg.norm(ball[:2]) < 5 and np.linalg.norm(v.ball_vel) < 1:
            ang = self._drive(v, ball, 2300, a)
            a[6] = 1.0 if me.boost > 0 else 0.0
            if np.linalg.norm(ball - me.pos) < 650 and abs(ang) < 0.2:
                self._start_flip(v, ball)
                self._continue_flip(v, a)
            return a

        k, sd = self._pick_intercept(v, pred_t, pred_p)
        p, t = pred_p[k], max(float(pred_t[k]), 1e-3)

        # ---------------------------------------------------------- rotation / defense
        goal_side = me.pos[1] < ball[1] - 100
        threat = ball[1] < -1500 or v.ball_vel[1] < -800
        opp_first = False
        if v.opp is not None and not v.opp.demoed:
            opp_first = np.linalg.norm(v.opp.pos - ball) + 400 < np.linalg.norm(me.pos - ball)
        if not goal_side and (threat or opp_first) and me.pos[1] > -4300:
            far_post = np.array([-np.sign(ball[0] + 1e-3) * 700.0, -GOAL_Y + 250, 17.0])
            # once level with the ball, cut back in towards the ball's line
            self._drive(v, far_post, 2300, a)
            return a

        # ---------------------------------------------------------- boost
        if pads is not None and me.boost < 25 and not threat:
            d_ball = np.linalg.norm(ball - me.pos)
            if d_ball > 2500:
                cand = [i for i in range(len(BIG_PADS)) if pads[i] > 0.5
                        and np.linalg.norm(BIG_PADS[i] - me.pos) < 2200 and BIG_PADS[i][1] < ball[1] + 500]
                if cand:
                    i = min(cand, key=lambda j: np.linalg.norm(BIG_PADS[j] - me.pos))
                    self._drive(v, BIG_PADS[i], 2300, a)
                    return a

        # ---------------------------------------------------------- attack the intercept
        rel = (me.pos - p)[:2]
        along = float(rel @ sd[:2])            # < 0: we are behind the ball (good side)
        perp = np.array([-sd[1], sd[0], 0.0])
        if along > -120:
            # wrong side of the ball for this shot: swing around it instead of pushing it backwards
            side = 1.0 if float(rel @ perp[:2]) >= 0 else -1.0
            target = p - sd * 650 + perp * side * 450
            speed = 2300.0
        else:
            target = p - sd * (BALL_RADIUS + 50)
            d = float(np.linalg.norm((target - me.pos)[:2]))
            to_p = _norm((p - me.pos) * np.array([1, 1, 0]))
            misalign = 0.5 * (1.0 - float(to_p @ sd))
            if d > 500:
                target = target - sd * min(0.45 * d, 800.0) * misalign
            speed = np.clip(d / t + 300.0, 700.0, 2300.0) if t > 0.25 else 2300.0
            speed = min(2300.0, speed * self.aggression)
        target = target.copy()
        target[0] = np.clip(target[0], -3900, 3900)
        target[1] = np.clip(target[1], -4950, 4950) if abs(target[0]) > 900 else np.clip(target[1], -5600, 5600)
        target[2] = 17.0
        ang = self._drive(v, target, speed, a)

        # ---------------------------------------------------------- hit: dodge into the ball
        to_ball = ball - me.pos
        dist_ball = float(np.linalg.norm(to_ball))
        fwd_speed = float(me.vel @ me.fwd)
        ball_ang = abs(self._angle_to(v, ball))
        good_dir = float(_norm(to_ball * np.array([1, 1, 0])) @ sd) > 0.55
        if (me.has_flip and good_dir and dist_ball < 380 and ball[2] < 220 and ball_ang < 0.5
                and fwd_speed > 700 and me.up[2] > 0.8):
            self._start_flip(v, ball + sd * 60)
            self._continue_flip(v, a)
        elif (me.has_flip and 200 < ball[2] < 420 and np.linalg.norm(to_ball[:2]) < 260
              and ball_ang < 0.6 and me.up[2] > 0.8):
            a[5] = 1.0        # jump to reach a bouncing ball
        return a
