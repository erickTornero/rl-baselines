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
from .losses import PPOLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
import rl_baselines
from rl_baselines.systems.base import RLBaseSystem
from .modules import AdvantageEstimator, TargetEstimator, TDError
import math
from torchrl.objectives.value import GAE

@rl_baselines.register("ppo-discrete")
class PPODiscreteSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        value_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.action_probs_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
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

        self.td_error_module = TensorDictModule(
            TDError(value_network, cfg.system.gamma),
            in_keys=[
                ('next', 'reward'),
                ('observation', ),
                ('next', 'observation'),
            ],
            out_keys=['td_error']
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
            CategoricalSampler(environment.action_spec, return_onehot=True),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy = TensorDictSequential(
            self.action_probs_module, 
            self.action_sampler,
        )

        self.loss_module = TensorDictModule(
            PPOLoss(self.cfg.system.epsilon),
            in_keys=['action_probs', 'action_probs_old', 'advantage', 'action', 'state_value', 'QTarget'],
            out_keys=['clip_loss', 'critic_loss']
        )
        self.automatic_optimization = False

    def forward(self, batch) -> TensorDict :
        return self.action_probs_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_bl] = self.optimizers()
        [scheduler_p, scheduler_v] = self.lr_schedulers()
        #import pdb;pdb.set_trace()
        # TODO: find a better way to parallelize it
        n_rollouts = self.cfg.data.nrollouts_per_iteration
        max_trajectory_length = self.cfg.data.max_trajectory_length
        buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=n_rollouts * max_trajectory_length, device=self.device),
            batch_size=self.cfg.data.batch_size
        )
        cum_rw = 0.0
        for iroll in range(n_rollouts):
            with torch.no_grad():
                episode_data = self.env.rollout(
                    max_trajectory_length,
                    policy=self.policy,
                    auto_cast_to_device=True
                )
                episode_data = self.td_error_module(episode_data)

            # compute Returns in linear time
            #import pdb;pdb.set_trace()
            self.gae(episode_data)
            #rewards = episode_data['next', 'reward']
            #episode_data['next', 'reward'] = (rewards - rewards.mean())/(rewards.std() + 1e-8)
            episode_data['advantage'] = (episode_data['advantage'] - episode_data['advantage'].mean())/(episode_data['advantage'].std() + 1e-8)
            """
            advantages = [
                torch.zeros((1, ), dtype=torch.float32, device=episode_data.device) 
                for _ in range(len(episode_data))
            ]
            advantages[-1] = episode_data[-1]['td_error']
            gae_gamma = self.cfg.system.gamma * self.cfg.system.gae
            #import pdb;pdb.set_trace()
            for t_episode in range(len(episode_data) - 2, -1, -1):
                advantage = advantages[t_episode + 1] * (gae_gamma) + episode_data[t_episode]['td_error']
                advantages[t_episode] = advantage

            import pdb;pdb.set_trace()
            episode_data['advantage'] = torch.vstack(advantages)
            """
            # setting old policy probabilities to compute r(theta)
            episode_data['action_probs_old'] = episode_data['action_probs'].clone()
            buffer.extend(episode_data)

            cum_rw += episode_data['next', 'reward'].sum()
        cum_rw /= n_rollouts
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        nbatches = math.ceil(self.cfg.trainer.epochs_per_episode * len(buffer)//self.cfg.data.batch_size)

        for ibatch in range(nbatches):
            tensordict = buffer.sample().clone()
            with torch.no_grad():
                tensordict = self.target_module(tensordict)
            tensordict = self.action_probs_module(tensordict)
            tensordict = self.value_module(tensordict)
            tensordict = self.loss_module(tensordict)

            optimizer.zero_grad()
            self.manual_backward(tensordict.get('clip_loss').mean())
            optimizer.step()

            optimizer_bl.zero_grad()
            self.manual_backward(tensordict.get('critic_loss').mean())
            optimizer_bl.step()
        scheduler_p.step()
        scheduler_v.step()


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
    def from_config(cls, config: Union[str, OmegaConf]) -> PPODiscreteSystem:
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

        state_value_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.state_value_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        return cls(cfg, policy_network, state_value_network, env)