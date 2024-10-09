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
from .losses import ReinforceContinuousWithBaselineLoss
import rl_baselines
from torchrl.data import ReplayBuffer, LazyTensorStorage
from rl_baselines.systems.base import RLBaseSystem
from torchrl.envs import EnvBase

@rl_baselines.register("reinforce-continuous-baseline")
class ReinforceContinuousWithBaselineSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        mean_network: Union[nn.Sequential, nn.Module],
        std_network: Union[nn.Sequential, nn.Module],
        baseline_network: Union[nn.Sequential, nn.Module],
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
        self.baseline_module = TensorDictModule(
            baseline_network,
            in_keys=['observation'],
            out_keys=['baseline']
        )

        self.action_sampler = TensorDictModule(
            ContinuousSampler(environment.action_spec),
            in_keys=['mean_action', 'std_action'],
            out_keys=['action']
        )

        self.policy_reinforce = TensorDictSequential(
            self.mean_module,
            self.std_module,
            self.baseline_module,
            self.action_sampler
        )
        self.loss_module = TensorDictModule(
            ReinforceContinuousWithBaselineLoss(self.cfg.system.gamma),
            in_keys=['mean_action', 'std_action', 'baseline', 'action', 'delta', 't'],
            out_keys=['pg_loss', 'baseline_loss']
        )


    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_bl] = self.optimizers()
        batch_size = self.cfg.data.batch_size
        epochs_per_episode = self.cfg.trainer.epochs_per_episode
        max_traj_len = self.cfg.data.max_trajectory_length
        with torch.no_grad():
            episode_data = self.env.rollout(
                max_traj_len,
                policy=self.policy_reinforce,
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

        deltas = torch.vstack(GL2) - episode_data['baseline']
        norm_deltas = (deltas - deltas.mean())/(deltas.std() + 1e-9)
        episode_data['delta'] = norm_deltas
        episode_data['t'] = torch.arange(0, len(episode_data)).unsqueeze(-1).to(episode_data.device)

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=max_traj_len, device=episode_data.device),
            batch_size=batch_size
        )
        ##import pdb;pdb.set_trace()

        replay_buffer.extend(episode_data)

        cum_rw = episode_data['next', 'reward'].sum()
        self.cum_rewards.append(cum_rw)
        avg_rw = sum(self.cum_rewards)/len(self.cum_rewards)

        nbatches_samples = (epochs_per_episode * len(episode_data)) // batch_size
        for ibatch in range(nbatches_samples):
            tensordict = replay_buffer.sample()
            probs_dict = self.mean_module(tensordict)
            probs_dict = self.std_module(tensordict)
            probs_dict = self.baseline_module(tensordict)
            #probs_dict = self.policy_reinforce(tensordict)
            loss_dict = self.loss_module(probs_dict)
            optimizer.zero_grad()
            self.manual_backward(loss_dict.get('pg_loss').mean())
            optimizer.step()

            optimizer_bl.zero_grad()
            self.manual_backward(loss_dict.get('baseline_loss').mean())
            optimizer_bl.step()
            ##distribution = Normal(probs_dict['mean'], probs_dict['std'])
            ##log_policy_action = distribution.log_prob(tensordict['action'])
            ##J = - log_policy_action * tensordict['G']
            ##loss = J.mean()
            ##optimizer.zero_grad()
            ##loss.backward()
            ##nn.utils.clip_grad_norm_(policy_network.parameters(), 1.0)
            ##optimizer.step()

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
                "Avg Episode Reward": avg_rw,
            }
        )

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(
            list(self.mean_module.parameters()) + list(self.std_module.parameters()), 
            **cfg_optimizer.args
        )
        optimizer_bl = optimizer_class(self.baseline_module.module.parameters(), **cfg_optimizer.args)

        return [optimizer, optimizer_bl]

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceContinuousWithBaselineSystem:
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

        baseline_network = RLBaseSystem.load_network_from_cfg(
            cfg.system.baseline_network,
            input_dim=get_env_obs_dim(env),
            output_dim=1,
        )

        return cls(cfg, mean_network, std_network, baseline_network, env)