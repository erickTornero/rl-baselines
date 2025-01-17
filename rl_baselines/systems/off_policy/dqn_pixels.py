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
        self.policy_explore = TensorDictSequential(
            self.action_values_module,
            self.egreedy_sampler_module
        )
        self.policy = TensorDictSequential(
            self.action_values_module,
            self.qsampler_module
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
        nfiring_actions = 0
        cumulator = Cummulator()

        for t in range(max_episode_length):
            if obs_dict['end-of-life'].item():
                # requires fire action
                action = self.env.action_spec.encode(self.fire_index)
                nfiring_actions += 1
                tensordict = TensorDict({'action': action, **obs_dict})
                tensordict = self.env.step(tensordict)
            else:
                with torch.no_grad():
                    #tensordict = self.policy_explore(obs_dict.unsqueeze(0))
                    tensordict = self.action_values_module(obs_dict.unsqueeze(0)).squeeze(0)
                    cumulator.step(tensordict['action_values'].max())
                    tensordict = self.egreedy_sampler_module(tensordict)

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

    def validation_step(self, batch):
        pass

    def test_rollout(self, save_video: bool=False):
        from torchrl.envs.transforms import EndOfLifeTransform, TransformedEnv
        from rl_baselines.utils.plot_stack import plot_rgb_stack
        #self.env = TransformedEnv(
        #    self.env,
        #    EndOfLifeTransform(
        #        eol_key="end-of-life",
        #        lives_key="lives",
        #        done_key="done",
        #    )
        #)
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

        if self.fire_index >= 0:
            td = self.env.step(
                TensorDict({
                    'action': self.env.action_spec.encode(self.fire_index),
                    **obs_dict
                })
            )

        pbar = tqdm(total=max_episode_steps)

        for istep in range(max_episode_steps):
            if obs_dict['end-of-life'].item():
                obs_dict['action'] = self.env.action_spec.encode(self.fire_index)
                action_dict = obs_dict
            else:
                with torch.no_grad():
                    #action_dict = self.policy(obs_dict)
                    action_dict = self.action_values_module(obs_dict.unsqueeze(0)).squeeze(0)
                    action_dict = self.qsampler_module(action_dict)
                    #print(action_dict['action'])
            next_dict = self.env.step(action_dict)
            #print(next_dict['next', 'truncated'].item(), next_dict['next', 'terminated'].item(), next_dict['next', 'end-of-life'].item())
            trajectory.append(next_dict)
            obs_dict = next_dict['next'].clone()
            reward = obs_dict.pop('reward')
            crw += reward
            done = next_dict['next', 'done']
            if render:
                img = self.env.render()
                #data_plot = plot_rgb_stack(next_dict['pixels'], next_dict['stack'], next_dict['next', 'stack'])
                #self.display_img(data_plot.cpu().numpy())
                self.display_img(self.env.render())
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
        k = cv2.waitKey(33)

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