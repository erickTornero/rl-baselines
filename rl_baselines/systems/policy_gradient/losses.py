import torch
from typing import Tuple, Union
from torch import nn
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

class ReinforceWithActorCriticLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, # must be with gradients
        state_value: torch.Tensor, # must be with gradients
        action: torch.Tensor,
        delta: torch.Tensor,
        gamma_cumm: Union[torch.Tensor, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = ReinforceLoss()(
            action_probs,
            action,
            delta,
        )
        reinforce_loss *= gamma_cumm

        critic_loss = -delta * state_value
        critic_loss = critic_loss.mean()

        return (reinforce_loss, critic_loss)

class BellmanDelta:
    def __init__(self, state_value_network: nn.Module, gamma: float) -> None:
        self.state_value_network = state_value_network
        self.gamma = gamma

    def __call__(
        self,
        observation: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_value = reward + self.gamma * self.state_value_network(next_observation) * (1 - done.to(dtype=torch.float32))
        expected_value = self.state_value_network(observation)
        delta = target_value - expected_value
        return delta, expected_value, target_value

class ReinforceContinuousLoss:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(
        self,
        mean_action: torch.Tensor,
        std_action: torch.Tensor,
        action: torch.Tensor,
        G: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.distributions.Normal(mean_action, std_action)
        log_p_action = dist.log_prob(action)
        gammm_t = (self.gamma)**(t)
        J = - gammm_t * log_p_action * G
        return J#.mean()

class ReinforceContinuousWithBaselineLoss(ReinforceContinuousLoss):
    def __init__(self, gamma: float) -> None:
        super().__init__(gamma)

    def __call__(
        self,
        mean_action: torch.Tensor,
        std_action: torch.Tensor,
        baseline: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = super().__call__(mean_action, std_action, action, delta, t)
        baseline_loss = -self.gamma * delta * baseline

        return (reinforce_loss, baseline_loss)

class ReinforceContinuousWithActorCriticLoss(ReinforceContinuousLoss):
    def __init__(self, gamma: float) -> None:
        super().__init__(gamma)

    def _reinforce_loss(
        self,
        mean_action: torch.Tensor,
        std_action: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
        gamma_cumm: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.distributions.Normal(mean_action, std_action)
        log_p_action = dist.log_prob(action)
        J = - gamma_cumm * delta * log_p_action 
        return J

    def __call__(
        self,
        mean_action: torch.Tensor,
        std_action: torch.Tensor,
        state_value: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
        gamma_cumm: torch.Tensor,
        expected_value: torch.Tensor,
        target_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = self._reinforce_loss(
            mean_action,
            std_action,
            action, 
            delta.detach(),
            gamma_cumm
        )
        critic_loss = nn.MSELoss()(target_value.detach(), expected_value)

        return (reinforce_loss, critic_loss)

from rl_baselines.common.distributions import NormalLogVar
class ReinforceContinuousLogVarLoss:
    def __call__(
        self,
        mean_action: torch.Tensor,
        logvar_action: torch.Tensor,
        maxlogvar_action: torch.Tensor,
        minlogvar_action: torch.Tensor,
        action: torch.Tensor,
        G: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        #dist = torch.distributions.Normal(mean_action, std_action)
        dist = NormalLogVar(mean_action, logvar_action, maxlogvar_action, minlogvar_action)
        log_p_action = dist.log_loss(action)
        #J = -0.99**(t) * log_p_action * G
        J = -log_p_action * G
        return J#.mean()