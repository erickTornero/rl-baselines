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
from .action_sampler import ContinuousSampler
from .losses import PPOContinuousLoss
import rl_baselines
from torchrl.data import ReplayBuffer, LazyTensorStorage
from rl_baselines.systems.base import RLBaseSystem
from torchrl.envs import EnvBase
from torchrl.objectives.value import GAE
import math
from .modules import TargetEstimator

@rl_baselines.register("ppo-continuous")
class PPOContinuousSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        mean_network: Union[nn.Sequential, nn.Module],
        std_network: Union[nn.Sequential, nn.Module],
        value_network: Union[nn.Sequential, nn.Module],
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
        self.value_module = TensorDictModule(
            value_network,
            in_keys=['observation'],
            out_keys=['state_value']
        )

        self.gae = GAE(
            gamma=cfg.system.gamma,
            lmbda=cfg.system.gae,
            value_network=self.value_module,
            differentiable=False
        )

        self.target_module = TensorDictModule(
            TargetEstimator(value_network, cfg.system.gamma),
            in_keys=[
                ('next', 'reward'),
                ('next', 'observation'),
            ],
            out_keys=['QTarget']
        )

        self.action_sampler = TensorDictModule(
            ContinuousSampler(environment.action_spec),
            in_keys=['mean_action', 'std_action'],
            out_keys=['action']
        )

        self.policy = TensorDictSequential(
            self.mean_module,
            self.std_module,
            self.action_sampler
        )
        self.loss_module = TensorDictModule(
            PPOContinuousLoss(self.cfg.system.epsilon),
            in_keys=['mean_action', 'mean_action_old', 'std_action', 'std_action_old', 'advantage', 'action', 'state_value', 'QTarget'],
            out_keys=['clip_loss', 'critic_loss']
        )


    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_bl] = self.optimizers()
        [scheduler_p, scheduler_v] = self.lr_schedulers()
        batch_size = self.cfg.data.batch_size
        epochs_per_episode = self.cfg.trainer.epochs_per_episode
        max_traj_len = self.cfg.data.max_trajectory_length
        n_rollouts = self.cfg.data.nrollouts_per_iteration
        normalize_advantages = self.cfg.system.normalize_advantages
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=n_rollouts * max_traj_len, device=self.device),
            batch_size=batch_size
        )
        cum_rw = 0.0
        for i_roll in range(n_rollouts):
            with torch.no_grad():
                episode_data = self.env.rollout(
                    max_traj_len,
                    policy=self.policy,
                    auto_cast_to_device=True
                )
                #import pdb;pdb.set_trace()
                self.gae(episode_data)
                #episode_data['advantage'] = (episode_data['advantage'] - episode_data['advantage'].mean())/(episode_data['advantage'].std() + 1e-8)
            episode_data['mean_action_old'] = episode_data['mean_action'].clone()
            episode_data['std_action_old'] = episode_data['std_action'].clone()
            replay_buffer.extend(episode_data)
            cum_rw += episode_data['next', 'reward'].sum()

        if normalize_advantages:
            replay_buffer['advantage'] = (replay_buffer['advantage'] - replay_buffer['advantage'].mean())/(replay_buffer['advantage'].std() + 1e-8)
        cum_rw /= n_rollouts
        nbatches = math.ceil(epochs_per_episode * len(replay_buffer)//self.cfg.data.batch_size)

        for ibatch in range(nbatches):
            tensordict = replay_buffer.sample().clone()
            with torch.no_grad():
                tensordict = self.target_module(tensordict)
            tensordict = self.mean_module(tensordict)
            tensordict = self.std_module(tensordict)
            tensordict = self.value_module(tensordict)

            #probs_dict = self.policy_reinforce(tensordict)
            loss_dict = self.loss_module(tensordict)
            optimizer.zero_grad()
            #import pdb;pdb.set_trace()
            self.manual_backward(loss_dict.get('clip_loss').mean())
            optimizer.step()

            optimizer_bl.zero_grad()
            self.manual_backward(loss_dict.get('critic_loss').mean())
            optimizer_bl.step()
        scheduler_p.step()
        scheduler_v.step()

        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw,
            }
        )

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(
            list(self.mean_module.parameters()) + list(self.std_module.parameters()), 
            **cfg_optimizer.args
        )
        optimizer_bl = optimizer_class(self.value_module.module.parameters(), **cfg_optimizer.args)

        if 'scheduler' in self.cfg.system:
            scheduler_class = getattr(optim.lr_scheduler, self.cfg.system.scheduler.name)
            scheduler_policy = scheduler_class(optimizer, **self.cfg.system.scheduler.args)
            scheduler_value = scheduler_class(optimizer_bl, **self.cfg.system.scheduler.args)
            return (
                {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler_policy,
                },
                {
                    "optimizer": optimizer_bl,
                    "lr_scheduler": scheduler_value
                }
            )
        return [optimizer, optimizer_bl]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> PPOContinuousSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)
        #import pdb;pdb.set_trace()
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

        value_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.value_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        return cls(cfg, mean_network, std_network, value_network, env)