import numpy as np
import torch

from flybrain.brain.controller import FlyController
from flybrain.brain.plasticity import DopaminePlasticity, PlasticityConfig
from flybrain.train.es import Genome, _centered_ranks, apply_population, es_units, fold_genome


def test_units_cover_every_neuron(graph):
    names, uid = es_units(graph, min_size=4)
    assert uid.shape == (graph.n,) and uid.max() == len(names) - 1


def test_population_members_get_their_own_gains(graph, brain):
    names, uid = es_units(graph, 4)
    pcfg = PlasticityConfig()
    genome = Genome(len(names), pcfg)
    plast = DopaminePlasticity(brain, graph, pcfg)
    ctrl = FlyController(brain, 4, plasticity=plast)
    pop = np.zeros((4, genome.size))
    pop[1, :genome.U] = 0.5              # member 1: stronger presynaptic release everywhere
    pop[2, 2 * genome.U] = np.log(2.0)   # member 2: double learning rate
    apply_population(brain, ctrl, genome, torch.from_numpy(uid), pop)
    gp, _ = brain.gains()
    assert torch.allclose(ctrl.g_pre[:, 0], gp)
    assert torch.allclose(ctrl.g_pre[:, 1], gp * np.exp(0.5), rtol=1e-5)
    assert np.isclose(float(plast.eta[2]), 2 * pcfg.eta, rtol=1e-4)


def test_fold_genome_matches_population_member(graph, brain):
    names, uid = es_units(graph, 4)
    genome = Genome(len(names), PlasticityConfig())
    mu = np.random.default_rng(0).normal(0, 0.1, genome.size)
    ctrl = FlyController(brain, 1)
    apply_population(brain, ctrl, genome, torch.from_numpy(uid), mu[None])
    g_pre_member = ctrl.g_pre[:, 0].clone()
    fold_genome(brain, genome, torch.from_numpy(uid), mu)
    gp, _ = brain.gains()
    assert torch.allclose(gp, g_pre_member, rtol=1e-5)


def test_centered_ranks():
    r = _centered_ranks(np.array([3.0, -1.0, 10.0, 0.0]))
    assert np.allclose(sorted(r), [-0.5, -1 / 6, 1 / 6, 0.5])
    assert r[2] == 0.5 and r[1] == -0.5
