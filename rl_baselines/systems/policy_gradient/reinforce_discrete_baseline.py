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
from .losses import ReinforceWithBaselineLoss

import rl_baselines
from rl_baselines.systems.base import RLBaseSystem

@rl_baselines.register("reinforce-discrete-baseline")
class ReinforceDiscreteBaselineSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        baseline_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.action_probs_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
        )
        self.baseline_module = TensorDictModule(
            baseline_network,
            in_keys=['observation'], 
            out_keys=['baseline_value']
        )
        self.action_sampler = TensorDictModule(
            CategoricalSampler(),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy = TensorDictSequential(self.action_probs_module, self.action_sampler)
        self.loss_module = TensorDictModule(
            ReinforceWithBaselineLoss(),
            in_keys=['action_probs', 'baseline_value', 'action', 'G'],
            out_keys=['reinforce_loss', 'baseline_loss']
        )
        self.automatic_optimization = False

    def forward(self, batch) -> TensorDict :
        return self.action_probs_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_bl] = self.optimizers()
        with torch.no_grad():
            episode_data = self.env.rollout(
                self.cfg.data.max_trajectory_length,
                policy=self.policy,
                auto_cast_to_device=True
            )
        # compute Returns in linear time
        GL2 = [
            torch.zeros((1, ), dtype=torch.float32, device=episode_data.device) 
            for _ in range(len(episode_data))
        ]
        for t_episode in range(len(episode_data) - 2, -1, -1):
            G = GL2[t_episode + 1] * self.cfg.system.gamma + episode_data[t_episode + 1]['next', 'reward']
            GL2[t_episode] = G

        episode_data['G'] = torch.vstack(GL2)

        cum_rw = episode_data['next', 'reward'].sum()
        #import pdb;pdb.set_trace()
        for t_episode in range(len(episode_data) - 1):
            tensordict = episode_data[t_episode]
            #action = tensordict['action']
            # uncomment following line to allow gymenv environment
            #action = tensordict['action'].argmax()
            tensordict = self.action_probs_module(tensordict)
            tensordict = self.baseline_module(tensordict)
            #J = torch.gather(probs_dict['action_probs'], 0, action)
            #J = -torch.log(J) * tensordict['G']
            losses = self.loss_module(tensordict)
            optimizer.zero_grad()
            self.manual_backward(losses.get('reinforce_loss'))
            optimizer.step()

            optimizer_bl.zero_grad()
            self.manual_backward(losses.get('baseline_loss'))
            optimizer_bl.step()

        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw
            }, 
        )

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(self.action_probs_module.module.parameters(), **cfg_optimizer.args)
        optimizer_bl = optimizer_class(self.baseline_module.module.parameters(), **cfg_optimizer.args)
        return [optimizer, optimizer_bl]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceDiscreteBaselineSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config

        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)

        policy_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.policy_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env),
        )

        baseline_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.baseline_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        return cls(cfg, policy_network, baseline_network, env)