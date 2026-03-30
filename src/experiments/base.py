from abc import ABC, abstractmethod


class Experiment(ABC):
    @abstractmethod
    def build_data(self, args, rank, world_size):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, args, meta):
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, args, model):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, model, batch, device):
        raise NotImplementedError

    def print_startup_info(self, dataset, meta, args):
        pass

    def generate_sample(self, model, dataset, device, args, global_step):
        pass