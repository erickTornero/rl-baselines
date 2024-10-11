from __future__ import annotations
from omegaconf import OmegaConf
from typing import Union
from torch import nn
from torchrl.envs import EnvBase
from rl_baselines.systems.base import RLBaseSystem
from tensordict.nn import TensorDictModule, TensorDictSequential
from .sampler import ContinuousExplorationDDPGSampler, ActionContinuousClamper
import torch
from .losses import QTargetEstimatorContinuous, DDPGCriticLoss, DDPGPolicyLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
import rl_baselines
from rl_baselines.utils.weights import SoftUpdate
from rl_baselines.common.custom_envs import get_env_action_dim, get_env_obs_dim
from torch import optim
from rl_baselines.utils.noise import OrnsteinUhlenbeck
from rl_baselines.common.networks import init_final_linear_layer

@rl_baselines.register("ddpg")
class DDPGSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        state_action_value_network: Union[nn.Sequential, nn.Module],
        state_action_value_target_network: Union[nn.Sequential, nn.Module],
        policy_network:Union[nn.Sequential, nn.Module],
        policy_target_network:Union[nn.Sequential, nn.Module],
        noise_process: OrnsteinUhlenbeck,
        environment: EnvBase,
    ):
        RLBaseSystem.__init__(self, cfg, environment)
        self.qnetwork = state_action_value_network
        self.state_action_values_module = TensorDictModule(
            lambda x, y: state_action_value_network(torch.concat((x, y), dim=-1)),
            in_keys=['observation', 'action'], 
            out_keys=['state_action_value']
        )

        self.clamper = TensorDictModule(
            ActionContinuousClamper(environment.action_spec),
            in_keys=['action'],
            out_keys=['action']
        )

        self.policy = TensorDictSequential(
            TensorDictModule(
                policy_network,
                in_keys=['observation'],
                out_keys=['action']
            ),
            self.clamper
        )

        self.action_exploration_sampler = TensorDictModule(
            ContinuousExplorationDDPGSampler(noise_process),
            in_keys=['action'],
            out_keys=['action']
        )

        self.policy_explore = TensorDictSequential(
            self.policy,
            self.action_exploration_sampler,
            self.clamper
        )
        self.target_estimator = TensorDictModule(
            QTargetEstimatorContinuous(
                state_action_value_target_network,
                policy_target_network,
                self.cfg.system.gamma,
            ),
            in_keys=[
                ('next', 'reward'),
                ('next', 'observation'),
                ('next', 'done')
            ],
            out_keys=['state_action_value_target']
        )
        self.soft_update_qnetworks = SoftUpdate(
            state_action_value_network, 
            state_action_value_target_network,
            **cfg.system.soft_update
        )

        self.soft_update_policy = SoftUpdate(
            policy_network, 
            policy_target_network,
            **cfg.system.soft_update
        )
        self.critic_loss = TensorDictModule(
            DDPGCriticLoss(),
            in_keys=["state_action_value", "state_action_value_target"],
            out_keys=["critic_loss"]
        )
        self.policy_loss = TensorDictModule(
            DDPGPolicyLoss(state_action_value_network),
            in_keys=["observation", "action"],
            out_keys=["policy_loss"]
        )

    def on_fit_start(self):
        super().on_fit_start()
        self.memory_replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.data.replay_buffer_size, device=self.device),
            batch_size=self.cfg.data.batch_size
        )
        self.action_exploration_sampler.module.noise_process.to(self.device, self.dtype)
        self.clamper.module.to(self.device)


    def training_step(self, batch, batch_idx):
        [optimizer, optimizer_critic] = self.optimizers()
        obs_dict = self.env.reset()

        cum_rw = 0
        max_episode_length = self.cfg.data.max_trajectory_length
        self.action_exploration_sampler.module.reset_noise()
        for t in range(max_episode_length):
            with torch.no_grad():
                tensordict = self.policy_explore(obs_dict)
                #if t %5: print(tensordict["action"])

            tensordict = self.env.step(tensordict)
            self.memory_replay.add(tensordict)
            done = tensordict['next', 'done']
            obs_dict = tensordict['next'].clone()
            reward = obs_dict.pop('reward')

            if len(self.memory_replay) > self.cfg.data.min_replay_buffer_size:
                # train step here
                #import pdb;pdb.set_trace()
                batch = self.memory_replay.sample().clone()
                with torch.no_grad():
                    batch = self.target_estimator(batch)
                batch = self.state_action_values_module(batch) # compute Q(si, ai)
                loss_dict = self.critic_loss(batch)

                optimizer_critic.zero_grad()
                self.manual_backward(loss_dict.get('critic_loss').mean())
                optimizer_critic.step()

                batch = self.policy(batch) # recompute action with gradient tracking
                loss_dict = self.policy_loss(batch)

                optimizer.zero_grad()
                self.manual_backward(loss_dict.get('policy_loss').mean())
                optimizer.step()

                self.soft_update_policy()
                self.soft_update_qnetworks()


            cum_rw += reward
            if done.item():
                break

        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw
            }
        )

    def get_optimizer_from_cfg(self, cfg_optimizer, model: nn.Module) -> torch.optim.optimizer.Optimizer:
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(
            model.parameters(), 
            **cfg_optimizer.args
        )
        return optimizer

    def configure_optimizers(self):
        cfg_optimizers = self.cfg.system.optimizer

        optimizer_policy = self.get_optimizer_from_cfg(cfg_optimizers.policy, self.policy.module)
        optimizer_critic = self.get_optimizer_from_cfg(cfg_optimizers.critic, self.qnetwork)
        return [optimizer_policy, optimizer_critic]
    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> DDPGSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = RLBaseSystem.load_env_from_cfg(config.system.environment)
        state_action_network = RLBaseSystem.load_network_from_cfg(
            config.system.state_action_network,
            input_dim=get_env_obs_dim(env) + get_env_action_dim(env),
            output_dim=1
        )
        state_action_target_network = RLBaseSystem.load_network_from_cfg(
            config.system.state_action_network,
            input_dim=get_env_obs_dim(env) + get_env_action_dim(env),
            output_dim=1
        )

        policy_network = RLBaseSystem.load_network_from_cfg(
            config.system.policy_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env)
        )
        policy_target_network = RLBaseSystem.load_network_from_cfg(
            config.system.policy_network,
            input_dim=get_env_obs_dim(env),
            output_dim=get_env_action_dim(env)
        )

        if config.system.exploration.type == "ornstein":
            noise_process = OrnsteinUhlenbeck(
                **config.system.exploration.args,
                size=get_env_action_dim(env)
            )
        else:
            raise NotImplementedError(f"noise process <{config.system.exploration.type}> not supported")

        # initialize last layers according to the paper
        if 'from_pixels' in config.system.environment and config.system.environment.from_pixels:
            # for pixels case
            amplitude_last_layer = 3e-4
        else:
            amplitude_last_layer = 3e-3
        init_final_linear_layer(state_action_network, -amplitude_last_layer, amplitude_last_layer)
        init_final_linear_layer(policy_network, -amplitude_last_layer, amplitude_last_layer)

        return cls(
            cfg,
            state_action_network,
            state_action_target_network,
            policy_network,
            policy_target_network,
            noise_process,
            env,
        )