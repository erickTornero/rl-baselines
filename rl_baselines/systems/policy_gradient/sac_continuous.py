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
from .action_sampler import ContinuousReparametrizationTrickSampler
from .losses import SACValuesContinuousLoss, SACPolicyContinuousLoss
import rl_baselines
from torchrl.data import ReplayBuffer, LazyTensorStorage
from rl_baselines.systems.base import RLBaseSystem
from torchrl.envs import EnvBase
from torchrl.objectives.value import GAE
import math
from .modules import TargetEstimator
from rl_baselines.utils.weights import SoftUpdate

@rl_baselines.register("sac-continuous")
class SoftActorCriticContinuousSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        mean_network: Union[nn.Sequential, nn.Module],
        std_network: Union[nn.Sequential, nn.Module],
        value_network: Union[nn.Sequential, nn.Module],
        value_network_target: Union[nn.Sequential, nn.Module],
        state_action_value_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.state_action_value_network = state_action_value_network
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

        self.state_action_values_module = TensorDictModule(
            lambda x, y: state_action_value_network(torch.concat((x, y), dim=-1)),
            in_keys=['observation', 'action'], 
            out_keys=['state_action_value']
        )

        self.target_module = TensorDictModule(
            TargetEstimator(value_network_target, cfg.system.gamma),
            in_keys=[
                ('next', 'reward'),
                ('next', 'observation'),
            ],
            out_keys=['state_value_target']
        )

        self.soft_update_value_network = SoftUpdate(
            value_network,
            value_network_target,
            **cfg.system.soft_update
        )

        self.action_sampler = TensorDictModule(
            ContinuousReparametrizationTrickSampler(environment.action_spec),
            in_keys=['mean_action', 'std_action'],
            out_keys=['action']
        )

        self.policy = TensorDictSequential(
            self.mean_module,
            self.std_module,
            self.action_sampler
        )

        self.loss_values_module = TensorDictModule(
            SACValuesContinuousLoss(self.cfg.system.gamma),
            in_keys=[
                'mean_action',
                'std_action',
                'action',
                'state_value',
                'state_value_target',
                'state_action_value'
            ],
            out_keys=['values_loss', 'state_value_loss', 'state_action_value_loss']
        )

        self.loss_policy_module = TensorDictModule(
            SACPolicyContinuousLoss(cfg.system.gamma),
            in_keys=[
                'mean_action',
                'std_action',
                'action',
                'state_action_value'
            ],
            out_keys=["policy_loss"]
        )

    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def on_fit_start(self):
        super().on_fit_start()
        self.memory_replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.data.replay_buffer_size, device=self.device),
            batch_size=self.cfg.data.batch_size
        )

    def training_step(self, batch, batch_idx):
        [optimizer_policy, optimizer_values] = self.optimizers()
        obs_dict = self.env.reset()

        cum_rw = 0
        max_episode_length = self.cfg.data.max_trajectory_length
        for t in range(max_episode_length):
            with torch.no_grad():
                tensordict = self.policy(obs_dict)
                #if t %5: print(tensordict["action"])

            tensordict = self.env.step(tensordict)
            self.memory_replay.add(tensordict)
            done = tensordict['next', 'done']
            obs_dict = tensordict['next'].clone()
            reward = obs_dict.pop('reward')

            if len(self.memory_replay) > self.cfg.data.min_replay_buffer_size:
                # train step here
                batch = self.memory_replay.sample().clone()
                batch = self.mean_module(batch)
                batch = self.std_module(batch)
                batch = self.value_module(batch)
                batch = self.state_action_values_module(batch)
                with torch.no_grad():
                    batch = self.target_module(batch)
                
                loss_dict = self.loss_values_module(batch)

                optimizer_values.zero_grad()
                self.manual_backward(loss_dict.get('values_loss').mean())
                optimizer_values.step()
                batch = self.action_sampler(batch) # recompute action with gradient tracking
                batch = self.state_action_values_module(batch)
                loss_dict = self.loss_policy_module(batch)
                
                optimizer_policy.zero_grad()
                self.manual_backward(loss_dict.get('policy_loss').mean())
                #### detect high gradients

                for param in self.std_module.module.parameters():
                    if (torch.abs(param.grad).max() > 100.0).item():
                        import pdb;pdb.set_trace()
                    if torch.isnan(param.grad).any().item():
                        print("nan gradients")
                        import pdb;pdb.set_trace()
                ####
                optimizer_policy.step()

                for param in self.std_module.module.parameters():
                    if (torch.abs(param).max() > 10.0).item():
                        import pdb;pdb.set_trace()
                    if torch.isnan(param).any().item():
                        import pdb;pdb.set_trace()

                #self.soft_update_policy()
                self.soft_update_value_network()


            cum_rw += reward
            if done.item():
                break

        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw
            }
        )

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)

        optimizer_policy = optimizer_class(
            list(self.mean_module.parameters()) + list(self.std_module.parameters()),
            **cfg_optimizer.args
        )
        optimizer_values = optimizer_class(
            list(self.value_module.module.parameters()) + list(self.state_action_value_network.parameters()),
            **cfg_optimizer.args
        )
        #optimizer_state_action_value = optimizer_class(
        #    self.state_action_values_module.module.parameters(),
        #    **cfg_optimizer.args
        #)

        if 'scheduler' in self.cfg.system:
            scheduler_class = getattr(optim.lr_scheduler, self.cfg.system.scheduler.name)
            scheduler_policy = scheduler_class(optimizer_policy, **self.cfg.system.scheduler.args)
            scheduler_value = scheduler_class(optimizer_values, **self.cfg.system.scheduler.args)
            return (
                {
                    "optimizer": optimizer_policy,
                    "lr_scheduler": scheduler_policy,
                },
                {
                    "optimizer": optimizer_values,
                    "lr_scheduler": scheduler_value
                }
            )
        return [optimizer_policy, optimizer_values]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> SoftActorCriticContinuousSystem:
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

        value_network_target = RLBaseSystem.load_network_from_cfg(
            cfg.system.value_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        state_action_network = RLBaseSystem.load_network_from_cfg(
            config.system.state_action_network,
            input_dim=get_env_obs_dim(env) + get_env_action_dim(env),
            output_dim=1
        )

        return cls(cfg, mean_network, std_network, value_network, value_network_target, state_action_network, env)