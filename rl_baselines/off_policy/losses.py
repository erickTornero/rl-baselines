from torchrl.data import TensorSpec
import torch

class QTargetEstimator:
    def __init__(
        self,
        action_value_network: torch.nn.Module,
        gamma: float,
    ) -> None:
        self.gamma = gamma
        self.action_value_network = action_value_network

    def __call__(
        self, 
        reward: torch.Tensor, 
        next_observation: torch.Tensor, 
        next_done: torch.Tensor
    ) -> torch.Tensor:
        action_values = self.action_value_network(next_observation)
        notdone = 1.0 - next_done.to(dtype=torch.float32)
        return reward + self.gamma * torch.max(action_values, dim=-1, keepdim=True).values * notdone

class QLearningLoss:
    def __init__(
        self,
        use_onehot_actions: bool=False
    ) -> None:
        self.use_onehot_actions = use_onehot_actions
        self.mse_loss = torch.nn.MSELoss()

    def __call__(
        self, 
        action_value: torch.Tensor,
        action: torch.Tensor,
        target_action_values: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_onehot_actions:
            action_index = torch.argmax(action, dim=-1, keepdim=True)
        else:
            action_index = action
        chosen_action_values = torch.gather(action_value, -1, action_index)
        loss = self.mse_loss(target_action_values, chosen_action_values)
        loss = torch.sum((target_action_values - chosen_action_values)**2, dim=-1)
        return loss