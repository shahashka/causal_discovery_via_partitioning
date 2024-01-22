from abc import ABC, abstractmethod


class CausalDiscoveryAlgorithm(ABC):
    def __init__(self):
        pass

    @staticmethod
    def __call__(self):
        pass


class PartitioningAlgorithm(ABC):
    def __init__(self):
        pass

    @staticmethod
    def __call__(self):
        pass


class FusionAlgorithm(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class PC(CausalDiscoveryAlgorithm):
    def __init__(self, num_cores: int = 8):
        super().__init__()
        self.num_cores = 8

    def __call__(self):
        pass
