import os
import warnings

import pytest

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
os.environ.setdefault("FLYBRAIN_GENERATED_MESHES", "collision_meshes")


@pytest.fixture(scope="session")
def graph():
    from flybrain.connectome.synthetic import synthetic_graph
    return synthetic_graph()


@pytest.fixture()
def brain(graph):
    import torch
    from flybrain.brain.model import BrainConfig, FlyBrain
    torch.manual_seed(0)
    return FlyBrain(graph, BrainConfig())
