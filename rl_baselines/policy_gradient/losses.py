import torch
from typing import Tuple
class ReinforceLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, 
        action: torch.Tensor, 
        G: torch.Tensor
    ) -> torch.Tensor:
        J = torch.gather(action_probs, 0, action)
        J = -torch.log(J) * G
        return J.mean()

class ReinforceWithBaselineLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, # must be with gradients
        baseline_value: torch.Tensor, # must be with gradients
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = ReinforceLoss()(
            action_probs,
            action,
            delta,
        )

        baseline_loss = -delta * baseline_value
        baseline_loss = baseline_loss.mean()

        return (reinforce_loss, baseline_loss)