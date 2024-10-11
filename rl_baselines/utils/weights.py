# util functions to copy or make soft weights update in networks
from torch import nn
from typing import Optional
class SoftUpdate:
    def __init__(
        self,
        source_network: nn.Module,
        target_network: nn.Module,
        tau: float,
        initialize_same_weights: bool=True
    ):
        self.source_network = source_network
        self.target_network = target_network
        self.tau = tau
        if initialize_same_weights:
            self.init_same()
        assert tau > 0 and tau < 1.0, "tau must be in range <0, 1>"

    def __call__(self):
        self._soft_update()

    def _soft_update(self, tau: Optional[float]=None):
        tau = self.tau if tau is None else tau
        for tar_par, par in zip(self.target_network.parameters(), self.source_network.parameters()):
            tar_par.data.copy_(par.data * tau + tar_par.data * (1.0-tau))

    def init_same(self):
        self._soft_update(tau=1.0) # hard update