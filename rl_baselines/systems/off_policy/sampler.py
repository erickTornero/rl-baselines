import torch
from torch.distributions import Categorical
from torchrl.data import TensorSpec, OneHotDiscreteTensorSpec
class CategoricalSampler:
    def __init__(
        self, 
        action_spec: TensorSpec, 
        return_onehot: bool=False
    ) -> None:
        self.action_spec = action_spec
        self._return_onehot = return_onehot

    def __call__(self, action_probs: torch.Tensor) -> torch.Tensor:
        action = Categorical(action_probs).sample()
        if self._return_onehot and isinstance(self.action_spec, OneHotDiscreteTensorSpec):
            action = torch.nn.functional.one_hot(action, num_classes=self.action_spec.space.n)
        # uncomment following line to allow gymenv environment
        #action = torch.nn.functional.one_hot(action, num_classes=2)
        return action