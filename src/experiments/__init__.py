from .train_tiny_shakespeare.tiny_shakespeare_exp import TinyShakespeareExperiment


def get_experiment(name: str):
    if name == "tiny_shakespeare":
        return TinyShakespeareExperiment()

    raise ValueError(f"Unknown experiment: {name}")