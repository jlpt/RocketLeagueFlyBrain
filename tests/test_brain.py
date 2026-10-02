import numpy as np
import torch

from flybrain.brain.controller import FlyController
from flybrain.brain.model import N_ACTIONS, FlyBrain, load_brain, save_brain
from flybrain.brain.plasticity import DopaminePlasticity, PlasticityConfig, plastic_edges
from flybrain.connectome.graph import BrainGraph
from flybrain.sim.obs import OBS_DIM


def test_graph_roundtrip(graph, tmp_path):
    p = tmp_path / "g.npz"
    graph.save(p)
    g2 = BrainGraph.load(p)
    assert g2.n == graph.n and g2.nnz == graph.nnz
    assert np.array_equal(g2.indices, graph.indices)
    assert set(g2.groups) == set(graph.groups)


def test_step_shapes_and_bounds(brain):
    st = brain.init_state(3)
    obs = torch.randn(3, OBS_DIM)
    for _ in range(5):
        st, logits = brain.step(st, obs)
    assert logits.shape == (3, N_ACTIONS)
    assert torch.all(st["r"] >= 0) and torch.all(st["r"] < 1)
    a = FlyBrain.logits_to_action(logits)
    assert torch.all(a[:, :5].abs() <= 1) and set(a[:, 5:].unique().tolist()) <= {0.0, 1.0}


def test_dale_law_signs_fixed(graph, brain):
    w = brain.w_values.numpy()
    assert np.all(np.sign(w) == graph.sign[graph.indices])
    # learned gains are positive, so effective synapse signs can never flip
    gp, gq = brain.gains()
    assert torch.all(gp > 0) and torch.all(gq > 0)


def test_gradients_reach_neuron_params(brain):
    st = brain.init_state(2)
    obs = torch.randn(2, OBS_DIM)
    loss = 0
    for _ in range(6):
        st, logits = brain.step(st, obs)
        loss = loss + logits.pow(2).mean()
    loss.backward()
    for p in (brain.theta_pre, brain.theta_post, brain.bias, brain.tau_raw, brain.w_in["in_visual"]):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_custom_spmm_backward_matches_dense(brain):
    from flybrain.brain.model import spmm
    x = torch.randn(brain.n, 3, requires_grad=True)
    y = spmm(brain._W, brain._WT, x)
    y.pow(2).sum().backward()
    dense = brain._W.to_dense()
    x2 = x.detach().clone().requires_grad_(True)
    (dense @ x2).pow(2).sum().backward()
    assert torch.allclose(y, dense @ x.detach(), atol=1e-5)
    assert torch.allclose(x.grad, x2.grad, atol=1e-4)


def test_reaction_time_floor(brain):
    """No control change can appear before sensory delay + 1 hop + motor delay."""
    c = brain.cfg
    floor_steps = c.sensory_delay + 1 + c.motor_delay
    ctrl = FlyController(brain, 1)
    a_obs = np.zeros((1, OBS_DIM), np.float32)
    for _ in range(40):
        base = ctrl.step(a_obs)
    b_obs = np.random.default_rng(0).normal(size=(1, OBS_DIM)).astype(np.float32) * 3
    outs = [ctrl.step(b_obs) for _ in range(floor_steps + 6)]
    for k in range(floor_steps):
        assert np.allclose(outs[k][:, :5], base[:, :5], atol=1e-6), f"responded after {k} steps"
    assert any(not np.allclose(o[:, :5], base[:, :5], atol=1e-4) for o in outs[floor_steps:])


def test_checkpoint_roundtrip(graph, brain, tmp_path):
    with torch.no_grad():
        brain.theta_post += 0.1
    save_brain(tmp_path / "b.pt", brain, {"stage": "test"})
    b2, extra = load_brain(tmp_path / "b.pt", graph)
    assert extra["stage"] == "test"
    assert torch.allclose(b2.theta_post, brain.theta_post)
    st1, l1 = brain.step(brain.init_state(1), torch.ones(1, OBS_DIM))
    st2, l2 = b2.step(b2.init_state(1), torch.ones(1, OBS_DIM))
    assert torch.allclose(l1, l2, atol=1e-5)


def test_plasticity_respects_dale_and_learns(graph, brain):
    pcfg = PlasticityConfig(eta=5.0)
    plast = DopaminePlasticity(brain, graph, pcfg)
    assert plast.n_synapses == len(plastic_edges(graph))
    ctrl = FlyController(brain, 2, plasticity=plast)
    rng = np.random.default_rng(0)
    for t in range(60):
        obs = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
        ctrl.step(obs, reward=np.array([1.0 if t % 7 == 0 else 0.0, -1.0 if t % 5 == 0 else 0.0], np.float32))
    dW = plast.dW
    assert dW.abs().sum() > 0, "dopamine should change plastic synapses"
    assert torch.all(dW.abs() <= plast.w0[:, None] + 1e-6), "a synapse must never flip sign"
