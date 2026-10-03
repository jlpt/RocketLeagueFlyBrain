import neat
import numpy as np
import torch

from flybrain.train.hyperneat import N_INPUTS, PopulationSynapses, cppn_eval, edge_inputs, make_config


def _mutated_genomes(config, n=6, rounds=12):
    pop = neat.Population(config, seed=3)
    genomes = list(pop.population.values())[:n]
    for g in genomes:
        for _ in range(rounds):
            g.mutate(config.genome_config)
    return genomes


def test_vectorised_cppn_matches_neat(tmp_path):
    config = make_config(tmp_path / "neat.cfg", pop=8)
    X = np.random.default_rng(0).uniform(-1, 1, (50, N_INPUTS)).astype(np.float32)
    for g in _mutated_genomes(config):
        net = neat.nn.FeedForwardNetwork.create(g, config)
        ref = np.array([net.activate(row.tolist())[0] for row in X])
        assert np.allclose(cppn_eval(g, config, X), ref, atol=1e-4)


def test_edge_inputs_shape_and_mirror_features(graph):
    g = graph
    g.pos = np.random.default_rng(1).uniform(-1, 1, (g.n, 3)).astype(np.float32)
    X = edge_inputs(g)
    assert X.shape == (g.nnz, N_INPUTS)
    assert np.all(X[:, 0] >= 0) and np.all(X[:, 6] >= 0)       # |u| is mirror-symmetric
    assert set(np.unique(X[:, 12])) <= {-1.0, 0.0, 1.0}        # same-side indicator


def test_population_synapses_apply_each_members_weights(brain):
    P = 3
    syn = PopulationSynapses(brain, P)
    gains = np.random.default_rng(0).uniform(0.5, 2.0, (P, brain.col.numel())).astype(np.float32)
    syn.set_gains(gains)
    x = torch.rand(brain.n, P)
    y = syn(x)
    for p in range(P):
        Wp = torch.sparse_csr_tensor(brain.crow, brain.col, brain.w_values * torch.from_numpy(gains[p]),
                                     size=(brain.n, brain.n))
        assert torch.allclose(y[:, p], (Wp @ x[:, p:p + 1])[:, 0], atol=1e-5)
