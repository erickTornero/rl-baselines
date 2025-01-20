from __future__ import annotations
from typing import Union
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tqdm import tqdm
from rl_baselines.common import get_env_action_dim
from torchrl.envs import EnvBase
from .losses import QLearningLoss, QTargetEstimator
from rl_baselines.systems.base import RLBaseSystem
import cv2
import rl_baselines
from .egreedy import QPolicyExplorationSampler, QPolicySampler
from torchrl.data import ReplayBuffer, LazyTensorStorage
from rl_baselines.common.firing import FiringEndOfLifePolicy

class TransformAndScale:
    def __init__(self, scale: float=255.0, dtype: torch.dtype=torch.float32):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x/self.scale).to(dtype=self.dtype)
    
class Cummulator:
    def __init__(self):
        self.value = 0
        self.counter = 0

    def step(self, value):
        self.value += value
        self.counter += 1

    def get(self):
        return self.value
    
    def reset(self):
        self.value = 0
        self.counter = 0

    def mean(self):
        return self.value/self.counter

@rl_baselines.register("dqn-pixels")
class DQNDiscretePixelsSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        qvalues_network: Union[nn.Sequential, nn.Module],
        qvalues_network_target: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.action_spec = environment.action_spec
        self.action_values_module = TensorDictSequential(
            TensorDictModule(
                TransformAndScale(scale=255.0, dtype=torch.float32),
                in_keys=['observation'],
                out_keys=['stack_proc']
            ),
            TensorDictModule(
                qvalues_network,
                in_keys=['stack_proc'],
                out_keys=['action_values']
            )
        )

        self.loss_module = TensorDictModule(
            QLearningLoss(use_onehot_actions=True),
            in_keys=['action_values', 'action', 'Qtarget'],
            out_keys=['qloss']
        )

        self.egreedy_sampler_module = TensorDictModule(
            QPolicyExplorationSampler(
                self.action_spec,
                **self.cfg.system.egreedy,
                return_onehot=True # hardcoded
            ),
            in_keys=['action_values'],
            out_keys=['action', 'Qvalue']
        )
        self.qsampler_module = TensorDictModule(
            QPolicySampler(
                self.action_spec,
                return_onehot=True
            ),
            in_keys=['action_values'],
            out_keys=['action', 'Qvalue']
        )
        self.firing_module = TensorDictModule(
            FiringEndOfLifePolicy(self.env),
            in_keys=["action", "end-of-life"],
            out_keys=["action"]
        )
        self.policy_explore = TensorDictSequential(
            self.action_values_module,
            self.egreedy_sampler_module,
            self.firing_module
        )
        self.policy = TensorDictSequential(
            self.action_values_module,
            self.qsampler_module,
            self.firing_module
        )
        self.target_estimator = TensorDictSequential(
            TensorDictModule(
                TransformAndScale(scale=255.0, dtype=torch.float32),
                in_keys=[('next', 'observation')],
                out_keys=[('next', 'stack_proc')]
            ),
            TensorDictModule(
                QTargetEstimator(qvalues_network_target, self.cfg.system.gamma),
                in_keys=[
                    ('next', 'reward'),
                    ('next', 'stack_proc'),
                    ('next', 'done')
                ],
                out_keys=['Qtarget']
            )
        )

        self.update_target = self.load_update_networks(
            qvalues_network,
            qvalues_network_target,
            self.cfg.system.update_networks
        )

        self.gradient_update_counter = 0
        self.agent_selection_counter = 0
        self.total_frames = 0
        self.fire_index = self.env.unwrapped.get_action_meanings().index('FIRE') if 'FIRE' in self.env.unwrapped.get_action_meanings() else -1

        self.replay_keys =[
            'observation',
            'action',
            ('next', 'reward'),
            ('next', 'observation'),
            ('next', 'done'),
        ]

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)
        self.target_estimator[1].module.action_value_network.to(self.device)
        self.memory_replay = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.data.replay_buffer_size, device=self.device),
            batch_size=self.cfg.data.batch_size
        )

        #TODO: improve this way to send action spec to device
        self.action_spec = self.action_spec.to(self.device)
        self.qsampler_module.module.action_spec = self.action_spec
        self.egreedy_sampler_module.module.action_spec = self.action_spec


    def forward(self, batch) -> TensorDict :
        return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        optimizer = self.optimizers()
        obs_dict = self.env.reset()
        cum_rw = 0
        max_episode_length = self.cfg.data.max_trajectory_length
        log_qloss = 0
        cumulator = Cummulator()

        for t in range(max_episode_length):
            with torch.no_grad():
                tensordict = self.policy_explore(obs_dict.unsqueeze(0)).squeeze(0)
                cumulator.step(tensordict['action_values'].max())

            tensordict = self.env.step(tensordict)
            self.agent_selection_counter += 1

            if ((self.agent_selection_counter % self.cfg.system.grad_update_frequency == 0) \
                    and (len(self.memory_replay) > self.cfg.data.min_replay_buffer_size)):
                # train step here
                batch = self.memory_replay.sample().clone()
                with torch.no_grad():
                    batch = self.target_estimator(batch)
                batch = self.action_values_module(batch)
                loss_dict = self.loss_module(batch)
                optimizer.zero_grad()
                self.manual_backward(loss_dict.get('qloss').mean())
                with torch.no_grad():
                    log_qloss = loss_dict.get('qloss').mean()
                optimizer.step()
                self.egreedy_sampler_module.module.step_egreedy()
                self.gradient_update_counter += 1

                self.update_target()

                    #reward = obs_dict.pop('reward')
            obs_dict = tensordict["next"].clone()
            self.memory_replay.add(tensordict.select(*self.replay_keys).clone())

            reward = obs_dict.pop('reward')
            cum_rw += reward
            done = tensordict['next', 'done']
            self.total_frames += 1

            if done.item():
                break
        
        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw,
                "Replay Buffer Len": len(self.memory_replay),
                "qnetwork_loss": log_qloss,
                "Epsilon egreedy": self.egreedy_sampler_module.module.epsilon,
                "Action Value Avg": cumulator.mean(),
                "Total Gradient Updates": self.gradient_update_counter,
                "Total Agent Selections": self.agent_selection_counter,
                "Total Target Updates": self.update_target.nupdates,
                "Total Frames": self.total_frames,
                "Episode frames": t
            }
        )

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(self.action_values_module.module.parameters(), **cfg_optimizer.args)
        return optimizer

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> DQNDiscretePixelsSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = RLBaseSystem.load_env_from_cfg(cfg.system.environment)
        qnetwork = RLBaseSystem.load_network_from_cfg(
            cfg.system.network,
            0,
            get_env_action_dim(env)
        )
        qnetwork_target = RLBaseSystem.load_network_from_cfg(
            cfg.system.network,
            0,
            get_env_action_dim(env)
        )
        return cls(cfg, qnetwork, qnetwork_target, env)