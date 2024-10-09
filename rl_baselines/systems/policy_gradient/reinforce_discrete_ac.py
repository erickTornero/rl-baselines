from __future__ import annotations
from typing import Union
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from rl_baselines.common import (
    get_env_obs_dim, 
    get_env_action_dim
)
from torchrl.envs import EnvBase
from .action_sampler import CategoricalSampler
from .losses import ReinforceWithActorCriticLoss, BellmanDelta
import rl_baselines
from rl_baselines.systems.base import RLBaseSystem

@rl_baselines.register("reinforce-discrete-actor-critic")
class ReinforceDiscreteActorCriticSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        critic_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.action_probs_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
        )
        self.critic_module = TensorDictModule(
            critic_network,
            in_keys=['observation'], 
            out_keys=['state_value']
        )
        self.action_sampler = TensorDictModule(
            CategoricalSampler(),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy = TensorDictSequential(self.action_probs_module, self.action_sampler)
        self.loss_module = TensorDictModule(
            ReinforceWithActorCriticLoss(),
            in_keys=['action_probs', 'state_value', 'action', 'delta_bellman', 'gamma_cumm'],
            out_keys=['reinforce_loss', 'critic_loss']
        )
        self.bellman_delta_module = TensorDictModule(
            BellmanDelta(critic_network, self.cfg.system.gamma),
            in_keys=[
                ('observation',),
                ('next', 'reward'),
                ('next', 'observation'),
                ('next', 'done')
            ],
            out_keys=['delta_bellman', 'unused1', 'unused2']
        )

    def forward(self, batch) -> TensorDict :
        return self.action_probs_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_critic] = self.optimizers()

        gamma_cumm = 1
        cum_rw = 0.0
        obs_dict = self.env.reset()
        for istep in range(self.cfg.data.max_trajectory_length):
            with torch.no_grad():
                self.policy.eval()
                action_dict = self.policy(obs_dict)
                self.policy.train()
            next_dict = self.env.step(action_dict)
            with torch.no_grad():
                delta_dict = self.bellman_delta_module(next_dict.clone())

            tensordict = self.critic_module(delta_dict)
            tensordict = self.action_probs_module(tensordict)
            tensordict['gamma_cumm'] = gamma_cumm
            loss_dict = self.loss_module(tensordict)

            optimizer.zero_grad()
            self.manual_backward(loss_dict.get('reinforce_loss'))
            optimizer.step()
            optimizer_critic.zero_grad()
            self.manual_backward(loss_dict.get('critic_loss'))
            optimizer_critic.step()



            gamma_cumm *= self.cfg.system.gamma

            cum_rw += next_dict['next'].pop('reward')
            done = next_dict['next', 'done']
            
            obs_dict = next_dict['next'].clone()
            if done.item():
                break

        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw,
                "reinforce_loss": loss_dict.get('reinforce_loss'),
                "critic_loss": loss_dict.get('critic_loss')
            }, 
        )

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(self.action_probs_module.module.parameters(), **cfg_optimizer.args)
        optimizer_critic = optimizer_class(self.critic_module.module.parameters(), **cfg_optimizer.args)
        return [optimizer, optimizer_critic]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceDiscreteActorCriticSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config

        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)
        policy_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.policy_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env)
        )

        value_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.value_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1
        )

        return cls(cfg, policy_network, value_network, env)