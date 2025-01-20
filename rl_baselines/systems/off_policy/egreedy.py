from typing import Tuple, Union
import random
import torch
from torchrl.data import TensorSpec, OneHotDiscreteTensorSpec

class QPolicySampler:
    def __init__(
        self,
        action_spec: TensorSpec, 
        return_onehot: bool=False
    ) -> None:
        super().__init__()
        self.action_spec = action_spec
        self._return_onehot = return_onehot

    def sample_action(self, action_values: torch.Tensor) -> torch.Tensor:
        action = action_values.argmax(dim=-1)
        action = self._cast_action(action)
        return action

    def __call__(self, action_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.sample_action(action_values)
        if self._return_onehot:
            action_index = action.argmax(dim=-1, keepdim=True)
        else:
            action_index = action
        chosen_qvalues = torch.gather(action_values, -1, action_index)
        return (action, chosen_qvalues)
    
    def _cast_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._return_onehot and isinstance(self.action_spec, OneHotDiscreteTensorSpec):
            action = torch.nn.functional.one_hot(action, num_classes=self.action_spec.space.n)
        return action
    
    def to(self, device: str):
        self.action_spec = self.action_spec.to(device)
        return self

class QPolicyExplorationSampler(QPolicySampler):
    def __init__(
        self,
        action_spec: TensorSpec,
        epsilon_init: float,
        epsilon_end: float,
        annealing_steps: int,
        return_onehot: bool=False,
    ) -> None:
        super(QPolicyExplorationSampler, self).__init__(action_spec, return_onehot=return_onehot)
        self.decay_rate =   (-epsilon_end + epsilon_init)/annealing_steps
        self.epsilon = epsilon_init
        self.epsilon_end = epsilon_end

    def __call__(self, action_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ndim = action_values.ndim
        if random.random() < self.epsilon:
            if ndim == 1:
                action = self.action_spec.sample()
            elif ndim == 2:
                action = self.action_spec.sample((action_values.shape[0], ))
            else:
                raise NotImplementedError("")
            action_index = action.argmax(dim=-1, keepdim=True)
            chosen_qvalues = torch.gather(action_values, -1, action_index)
            if not self._return_onehot:
                action = action_index
        else:
            action, chosen_qvalues = super().__call__(action_values)
        return (action, chosen_qvalues)

    def step_egreedy(self):
        self.epsilon = max( self.epsilon_end, self.epsilon - self.decay_rate )
