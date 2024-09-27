import torch
from torch.distributions import Categorical
class CategoricalSampler:
    def __call__(self, action_probs: torch.Tensor) -> torch.Tensor:
        action = Categorical(action_probs).sample()
        # uncomment following line to allow gymenv environment
        #action = torch.nn.functional.one_hot(action, num_classes=2)
        return action