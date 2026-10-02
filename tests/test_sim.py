import math
from types import SimpleNamespace as NS

import numpy as np
import RocketSim as rs

from flybrain.play.packet import euler_to_rot, view_from_packet
from flybrain.sim import arena_mesh
from flybrain.sim.agents import IdleAgent, TeacherAgent
from flybrain.sim.env import Match
from flybrain.sim.meshes import mesh_hash, read_cmf
from flybrain.sim.obs import OBS_DIM, build_obs


def test_generated_mesh_hash_matches_rocketsim_cpp(tmp_path):
    # value verified against RocketSim's C++ CollisionMeshFile::UpdateHash
    out = arena_mesh.generate(tmp_path / "cm")
    assert hex(mesh_hash(*read_cmf(out / "soccar" / "flybrain_walls.cmf"))) == "0x26e7fc93"


def test_goal_is_scored_in_generated_arena():
    m = Match(seed=0)
    m.reset("kickoff")
    b = m.arena.ball.get_state()
    b.pos = rs.Vec(0, 4500, 100)
    b.vel = rs.Vec(0, 2500, 0)
    m.arena.ball.set_state(b)
    for _ in range(60):
        _, info = m.step([np.zeros(8), np.zeros(8)])
        if info["goal"] is not None:
            break
    assert info["goal"] == 0  # blue scores in the orange (+y) goal


def test_ball_bounces_off_back_wall_beside_goal():
    m = Match(seed=0)
    m.reset("kickoff")
    b = m.arena.ball.get_state()
    b.pos = rs.Vec(2000, 4500, 100)
    b.vel = rs.Vec(0, 2000, 0)
    m.arena.ball.set_state(b)
    ys = []
    for _ in range(30):
        views, info = m.step([np.zeros(8), np.zeros(8)])
        ys.append(views[0].ball_pos[1])
        assert info["goal"] is None
    assert max(ys) < 5120


def test_team_frames_are_mirrored():
    m = Match(seed=3)
    blue, orange = m.reset("random")
    assert np.allclose(blue.ball_pos[:2], -orange.ball_pos[:2], atol=1e-3)
    assert np.allclose(blue.me.pos[:2], -orange.opp.pos[:2], atol=1e-3)


def test_teacher_touches_the_ball():
    m = Match(seed=1, max_seconds=60)
    views = m.reset("kickoff")
    t, idle = TeacherAgent(0), IdleAgent()
    touched = False
    for _ in range(30 * 15):
        views, info = m.step([t.act(views[0], m), idle.act(views[1], m)])
        if info["touch"][0]:
            touched = True
            break
    assert touched


def _fake_packet(arena, cars):
    def phys(state):
        ang = state.rot_mat.as_angle()
        return NS(location=NS(**dict(zip("xyz", state.pos.as_tuple()))),
                  velocity=NS(**dict(zip("xyz", state.vel.as_tuple()))),
                  angular_velocity=NS(**dict(zip("xyz", state.ang_vel.as_tuple()))),
                  rotation=NS(pitch=ang.pitch, yaw=ang.yaw, roll=ang.roll))
    game_cars = []
    for c in cars:
        s = c.get_state()
        game_cars.append(NS(physics=phys(s), team=int(c.team), boost=s.boost, has_wheel_contact=s.is_on_ground,
                            jumped=s.has_jumped, double_jumped=s.has_double_jumped or s.has_flipped,
                            is_demolished=s.is_demoed))
    return NS(game_cars=game_cars, num_cars=len(cars), game_ball=NS(physics=phys(arena.ball.get_state())),
              game_info=NS(seconds_elapsed=0.0))


def test_euler_matches_rocketsim_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        yaw, pitch, roll = rng.uniform(-math.pi, math.pi), rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi)
        rm = rs.Angle(yaw, pitch, roll).as_rot_mat()
        R = euler_to_rot(pitch, yaw, roll)
        assert np.allclose(R[0], rm.forward.as_numpy(), atol=1e-4)
        assert np.allclose(R[1], rm.right.as_numpy(), atol=1e-4)
        assert np.allclose(R[2], rm.up.as_numpy(), atol=1e-4)


def test_rlbot_packet_gives_same_observation():
    m = Match(seed=5)
    m.reset("random")
    for _ in range(20):
        m.step([np.array([1, 0.3, 0, 0, 0, 0, 1, 0]), np.zeros(8)])
    views = m.views()
    pkt = _fake_packet(m.arena, m.cars)
    for idx in (0, 1):
        v_pkt = view_from_packet(pkt, idx, {})
        o1 = build_obs(views[idx], np.zeros(8))
        o2 = build_obs(v_pkt, np.zeros(8))
        assert o1.shape == (OBS_DIM,)
        # flip/jump availability is reconstructed differently; compare everything else
        assert np.allclose(o1[:45], o2[:45], atol=2e-3)
