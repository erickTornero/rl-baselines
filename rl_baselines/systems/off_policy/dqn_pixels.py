from __future__ import annotations
from typing import Union
from omegaconf import OmegaConf
import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from rl_baselines.common import get_env_action_dim
from torchrl.envs import EnvBase
from .losses import QTargetEstimator
from rl_baselines.systems.base import RLBaseSystem
import rl_baselines
from torchrl.data import ReplayBuffer, LazyTensorStorage
from rl_baselines.common.firing import FiringEndOfLifePolicy
from rl_baselines.common.metric_cummulator import Cummulator
from .qlearning import QLearningDiscreteSystem

class TransformAndScale:
    def __init__(self, scale: float=255.0, dtype: torch.dtype=torch.float32):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x/self.scale).to(dtype=self.dtype)

@rl_baselines.register("dqn-pixels")
class DQNDiscretePixelsSystem(QLearningDiscreteSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        qvalues_network: Union[nn.Sequential, nn.Module],
        qvalues_network_target: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        QLearningDiscreteSystem.__init__(self, cfg, qvalues_network, qvalues_network_target, environment)
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