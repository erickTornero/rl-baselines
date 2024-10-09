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
from .losses import ReinforceLoss
import rl_baselines
from rl_baselines.systems.base  import RLBaseSystem

@rl_baselines.register("reinforce-discrete")
class ReinforceDiscreteSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.policy_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
        )
        self.action_sampler = TensorDictModule(
            CategoricalSampler(),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy = TensorDictSequential(self.policy_module, self.action_sampler)
        self.loss_module = TensorDictModule(
            ReinforceLoss(),
            in_keys=['action_probs', 'action', 'G'],
            out_keys=['pg_loss']
        )

    def forward(self, batch) -> TensorDict :
        return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        optimizer = self.optimizers()
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
            probs_dict = self.policy_module(tensordict)
            #J = torch.gather(probs_dict['action_probs'], 0, action)
            #J = -torch.log(J) * tensordict['G']
            J = self.loss_module(probs_dict)
            optimizer.zero_grad()
            self.manual_backward(J.get('pg_loss'))
            optimizer.step()

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
        optimizer = optimizer_class(self.policy_module.module.parameters(), **cfg_optimizer.args)
        return optimizer

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceDiscreteSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)
        network = RLBaseSystem.load_network_from_cfg(
            cfg.system.network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env)
        )
        return cls(cfg, network, env)