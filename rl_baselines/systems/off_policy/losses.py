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

class QTargetEstimatorContinuous(torch.nn.Module):
    """
        From ddpg paper
    """
    def __init__(
        self,
        action_value_network_target: torch.nn.Module,
        policy_network_target: torch.nn.Module,
        gamma: float,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.action_value_network_target = action_value_network_target
        self.policy_network_target = policy_network_target

    def __call__(
        self,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        next_done: torch.Tensor
    ) -> torch.Tensor:
        next_action_target = self.policy_network_target(next_observation)
        assert next_action_target.ndim == next_observation.ndim
        next_state_action = torch.concatenate((next_observation, next_action_target), dim=-1)
        qvalues = self.action_value_network_target(next_state_action)
        notdone = 1.0 - next_done.to(dtype=torch.float32)
        return reward + self.gamma * qvalues * notdone

class DDPGCriticLoss:
    def __init__(self):
        self.mse_fn = torch.nn.MSELoss()

    def __call__(self, state_action_value: torch.Tensor, state_action_value_target: torch.Tensor):
        return torch.sum((state_action_value_target - state_action_value)**2, dim=-1)

class DDPGPolicyLoss:
    def __init__(self, state_action_value_network: torch.nn.Module):
        self.state_action_value_network = state_action_value_network

    def __call__(self, state: torch.tensor, action_policy: torch.Tensor):
        #self._deactivate_sa_gradients()
        Q = self.state_action_value_network(torch.concat((state, action_policy), dim=-1))
        #self._activate_sa_gradients()
        return -Q

    def _deactivate_sa_gradients(self):
        for p in self.state_action_value_network.parameters():
            p.requires_grad = False

    def _activate_sa_gradients(self):
        for p in self.state_action_value_network.parameters():
            p.requires_grad = True

class DDPGLoss:
    def __init__(self, qnetwork: torch.nn.Module):
        self.qnetwork = qnetwork

    def __call__(
        self,
        state_action_value: torch.Tensor,
        qtarget: torch.Tensor,
        policy_action: torch.Tensor
    ):
        pass