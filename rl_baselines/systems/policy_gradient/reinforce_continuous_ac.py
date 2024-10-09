from __future__ import annotations
from typing import Union
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from rl_baselines.common import (
    get_env_obs_dim,
    get_env_action_dim,
)
from torchrl.envs import EnvBase
from .action_sampler import ContinuousSampler
from .losses import ReinforceContinuousWithActorCriticLoss
import rl_baselines
from torchrl.data import BoundedTensorSpec, ReplayBuffer, LazyTensorStorage
from rl_baselines.systems.base import RLBaseSystem
from .losses import BellmanDelta
@rl_baselines.register("reinforce-continuous-ac")
class ReinforceContinuousWithActorCriticSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        mean_network: Union[nn.Sequential, nn.Module],
        std_network: Union[nn.Sequential, nn.Module],
        critic_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.mean_module = TensorDictModule(
            mean_network,
            in_keys=['observation'],
            out_keys=['mean_action']
        )

        self.std_module = TensorDictModule(
            std_network,
            in_keys=['observation'],
            out_keys=['std_action']
        )
        self.critic_module = TensorDictModule(
            critic_network,
            in_keys=['observation'],
            out_keys=['state_value']
        )

        self.action_sampler = TensorDictModule(
            ContinuousSampler(environment.action_spec),
            in_keys=['mean_action', 'std_action'],
            out_keys=['action']
        )

        self.policy = TensorDictSequential(
            self.mean_module,
            self.std_module,
            self.critic_module,
            self.action_sampler
        )
        self.loss_module = TensorDictModule(
            ReinforceContinuousWithActorCriticLoss(self.cfg.system.gamma),
            in_keys=['mean_action', 'std_action', 'state_value', 'action', 'delta_bellman', 'gamma_cumm', 'expected_value', 'target_value'],
            out_keys=['pg_loss', 'critic_loss']
        )

        self.bellman_delta_module = TensorDictModule(
            BellmanDelta(critic_network, self.cfg.system.gamma),
            in_keys=[
                ('observation',),
                ('next', 'reward'),
                ('next', 'observation'),
                ('next', 'done')
            ],
            out_keys=['delta_bellman', 'expected_value', 'target_value']
        )

    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_critic] = self.optimizers()
        batch_size = self.cfg.data.batch_size
        epochs_per_episode = self.cfg.trainer.epochs_per_episode
        max_traj_len = self.cfg.data.max_trajectory_length

        gamma_cumm = 1
        cum_rw = 0.0
        obs_dict = self.env.reset()
        for istep in range(max_traj_len):
            #with torch.no_grad():
            self.policy.eval()
            tensordict = self.policy(obs_dict)
            self.policy.train()
            next_dict = self.env.step(tensordict)
            #with torch.no_grad():
            tensordict = self.bellman_delta_module(next_dict.clone())
            #tensordict = self.critic_module(delta_dict)
            #tensordict = self.mean_module(tensordict)
            #tensordict = self.std_module(tensordict)
            tensordict['gamma_cumm'] = gamma_cumm
            loss_dict = self.loss_module(tensordict)

            optimizer.zero_grad()
            self.manual_backward(loss_dict.get('pg_loss'))
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


        ## #import pdb;pdb.set_trace()
        ## for t_episode in range(len(episode_data) - 1):
        ##     tensordict = episode_data[t_episode]
        ##     #action = tensordict['action']
        ##     # uncomment following line to allow gymenv environment
        ##     #action = tensordict['action'].argmax()
        ##     tensordict = self.mean_module(tensordict)
        ##     tensordict = self.std_module(tensordict)
        ##     #J = torch.gather(probs_dict['action_probs'], 0, action)
        ##     #J = -torch.log(J) * tensordict['G']
        ##     J = self.loss_module(tensordict)
        ##     optimizer.zero_grad()
        ##     self.manual_backward(J.get('pg_loss'))
        ##     optimizer.step()
        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw,
            }
        )

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(
            list(self.mean_module.parameters()) + list(self.std_module.parameters()), 
            **cfg_optimizer.args
        )
        optimizer_critic = optimizer_class(self.critic_module.module.parameters(), **cfg_optimizer.args)

        return [optimizer, optimizer_critic]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceContinuousWithActorCriticSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)

        mean_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.mean_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env)
        )

        std_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.std_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env),
        )

        critic_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.critic_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        return cls(cfg, mean_network, std_network, critic_network, env)