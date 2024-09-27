import torch

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