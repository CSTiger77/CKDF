import abc
from torch import nn


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.

    Adds methods for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC) and
    "synaptic intelligence" (SI) to its subclasses.'''

    def __init__(self):
        super().__init__()

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass
