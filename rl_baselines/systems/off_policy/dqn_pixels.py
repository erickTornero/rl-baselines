from __future__ import annotations
from typing import Union, List, Optional
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tqdm import tqdm
from rl_baselines.common import (
    cnn_dqn,
    make_custom_envs,
    get_env_obs_dim,
    get_env_action_dim
)
from rl_baselines.common.preprocessing import DQNPreprocessing, StackObservation
from rl_baselines.utils.save_utils import SaveUtils
from torchrl.envs import EnvBase
from .losses import QLearningLoss, QTargetEstimator
import pytorch_lightning as pl
from rl_baselines.systems.base import RLBaseSystem
import cv2
import rl_baselines
from .egreedy import QPolicyExplorationSampler, QPolicySampler
from torchrl.data import ReplayBuffer, LazyTensorStorage

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
class DQNDiscreteSystem(RLBaseSystem):
    def __init__(
        self,
        cfg: OmegaConf,
        qvalues_network: Union[nn.Sequential, nn.Module],
        qvalues_network_target: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
        preprocessor: Optional[DQNDiscreteSystem]=None,
        stack: Optional[StackObservation]=None,
    ) -> None:
        RLBaseSystem.__init__(self, cfg, environment)
        self.action_spec = environment.action_spec
        self.action_values_module = TensorDictSequential(
            TensorDictModule(
                TransformAndScale(scale=255.0, dtype=torch.float32),
                in_keys=['observation'],
                out_keys=['observation_proc']
            ),
            TensorDictModule(
                qvalues_network,
                in_keys=['observation_proc'], 
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
        self.policy_explore = TensorDictSequential(
            self.action_values_module,
            self.egreedy_sampler_module
        )
        self.policy = TensorDictSequential(
            self.action_values_module,
            self.qsampler_module
        )
        self.target_c = TensorDictModule(
            QTargetEstimator(qvalues_network_target, self.cfg.system.gamma),
            in_keys=[
                ('next', 'reward'),
                ('next', 'observation_proc'),
                ('next', 'done')
            ],
            out_keys=['Qtarget']
        )
        self.target_estimator = TensorDictSequential(
            TensorDictModule(
                TransformAndScale(scale=255.0, dtype=torch.float32),
                in_keys=[('next', 'observation')],
                out_keys=[('next', 'observation_proc')]
            ),
            self.target_c
            #TensorDictModule(
            #    QTargetEstimator(qvalues_network_target, self.cfg.system.gamma),
            #    in_keys=[
            #        ('next', 'reward'),
            #        ('next', 'observation'),
            #        ('next', 'done')
            #    ],
            #    out_keys=['Qtarget']
            #)
        )
        self.update_target = self.load_update_networks(
            qvalues_network,
            qvalues_network_target,
            self.cfg.system.update_networks
        )
        self.preprocessor = preprocessor
        self.stack = stack
        self.gradient_update_counter = 0

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
        self.stack.reset()
        observation_tensor = obs_dict['pixels']
        self.stack.push(self.preprocessor(observation_tensor))
        obs_dict['observation'] = self.stack.get()

        log_qloss = 0
        cumulator = Cummulator()

        for t in range(max_episode_length):
            if t % self.cfg.system.frame_skip == 0:
                with torch.no_grad():
                    #tensordict = self.policy_explore(obs_dict.unsqueeze(0))
                    tensordict = self.action_values_module(obs_dict.unsqueeze(0)).squeeze(0)
                    cumulator.step(tensordict['action_values'].max())
                    tensordict = self.egreedy_sampler_module(tensordict)

                tensordict = self.env.step(tensordict)
                # TODO: fix this, nested pop dont work
                new_pixels = tensordict['next', 'pixels']
                self.stack.push(
                    self.preprocessor(
                        new_pixels,
                        observation_tensor
                    )
                )
                observation_tensor = new_pixels
                last_action = tensordict['action']
                tensordict['next', 'observation'] = self.stack.get()
                self.memory_replay.add(tensordict.exclude('pixels', 'observation_proc',('next', 'pixels'), ('next', 'observation_proc')))

                done = tensordict['next', 'done']
                obs_dict = tensordict['next'].clone()
                reward = obs_dict.pop('reward')

                if len(self.memory_replay) > self.cfg.data.min_replay_buffer_size:
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

                cum_rw += reward
                self.update_target()
            else:
                tensordict = self.env.step(TensorDict({'action': last_action, 'observation': self.stack.get()}))
                #observation_tensor = tensordict['next', 'pixels']
                # TODO: check behavior
                self.stack.push(
                    self.preprocessor(
                        tensordict['next', 'pixels'],
                        observation_tensor
                    )
                )
                tensordict['next', 'observation'] = self.stack.get()
                observation_tensor = tensordict['next', 'pixels']
                self.memory_replay.add(tensordict.exclude('pixels', ('next', 'pixels')))
                obs_dict = tensordict["next"].clone()
                reward = obs_dict.pop('reward')
                cum_rw += reward


            if done.item():
                break
        
        self.log_dict_with_envname(
            {
                "Episode Reward": cum_rw,
                "Gradient Updates": self.gradient_update_counter,
                "Replay Buffer Len": len(self.memory_replay),
                "qnetwork_loss": log_qloss,
                "Epsilon egreedy": self.egreedy_sampler_module.module.epsilon,
                "Action Value Avg": cumulator.mean()
            }
        )

    def validation_step(self, batch):
        pass

    def test_rollout(self, save_video: bool=False):
        obs_dict = self.env.reset()
        render = self.env._env.render_mode == 'rgb_array'
        if render:
            img = self.env.render()
            self.display_img(img)
            if save_video:
                video = cv2.VideoWriter(
                    self.get_absolute_path('output.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    125,
                    (img.shape[1], img.shape[0]),
                    #False
                )
                video.write(img)
                #frames.append(img)
        trajectory = []
        crw = 0
        max_episode_steps = self.cfg.data.max_trajectory_length
        pbar = tqdm(total=max_episode_steps)
        for istep in range(max_episode_steps):
            with torch.no_grad():
                action_dict = self.policy(obs_dict)
            next_dict = self.env.step(action_dict)
            trajectory.append(next_dict)
            obs_dict = next_dict['next'].clone()
            reward = obs_dict.pop('reward')
            crw += reward
            done = next_dict['next', 'done']
            if render:
                img = self.env.render()
                self.display_img(img)
                if save_video:
                    #frames.append(img)
                    video.write(img)
            pbar.update(1)
            #import pdb;pdb.set_trace()
            if done.item():
                break
        pbar.close()
        if save_video:
            video.release()

        print(f"Episode finised at step: {istep + 1}/{max_episode_steps}, Episode Reward: {crw.item():.2f}")

    def display_img(self, img):
        cv2.imshow(f'Reinforce Discrete', img)
        k = cv2.waitKey(1)

    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(self.action_values_module.module.parameters(), **cfg_optimizer.args)
        return optimizer

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        print(f'Loading state dict from {path}')
        self.load_state_dict(ckpt['state_dict'])

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> DQNDiscreteSystem:
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
        if 'preprocess' in config.system and config.system.preprocess is not None:
            preprocessor = RLBaseSystem.load_preprocessor(config.system.preprocess)
        else:
            preprocessor = None
        if 'stack' in config.system and config.system.stack is not None:
            stack = RLBaseSystem.load_stackruntime(config.system.stack)
        else:
            stack = None
        
        return cls(cfg, qnetwork, qnetwork_target, env, preprocessor, stack)