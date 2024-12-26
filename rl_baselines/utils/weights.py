# util functions to copy or make soft weights update in networks
from torch import nn
from typing import Optional, Union

class UpdateNetworks:
    def __init__(
        self,
        source_network: Union[nn.Module, nn.Sequential],
        target_network: Union[nn.Module, nn.Sequential],
    ):
        self.source_network = source_network
        self.target_network = target_network

    def __call__(self, *args, **kwds):
        raise NotImplementedError("")

    def init_same(self):
        raise NotImplementedError("")

class SoftUpdate(UpdateNetworks):
    def __init__(
        self,
        source_network: nn.Module,
        target_network: nn.Module,
        tau: float,
        initialize_same_weights: bool=True
    ):
        super().__init__(source_network, target_network)
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


class HardUpdate(UpdateNetworks):
    def __init__(
        self,
        source_network: nn.Module,
        target_network: nn.Module,
        update_frequency: float,
        initialize_same_weights: bool=True
    ):
        super().__init__(source_network, target_network)
        self.update_frequency = update_frequency
        self.initialize_same_weights = initialize_same_weights
        self.counter = 0

    def __call__(self):
        if self.counter % self.update_frequency == 0:
            self._hard_update()
        self.counter += 1

    def _hard_update(self):
        for tar_par, par in zip(self.target_network.parameters(), self.source_network.parameters()):
            tar_par.data.copy_(par.data)

    def init_same(self):
        self._hard_update() # hard update