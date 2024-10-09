from __future__ import annotations
from typing import Union, List, Optional
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tqdm import tqdm
from rl_baselines.common import (
    mlp_builder, 
    make_custom_envs, 
    get_env_obs_dim, 
    get_env_action_dim
)
from rl_baselines.utils.save_utils import SaveUtils
from torchrl.envs import EnvBase
from .losses import QLearningLoss, QTargetEstimator
import pytorch_lightning as pl
import cv2
import rl_baselines
from .egreedy import QPolicyExplorationSampler, QPolicySampler
from torchrl.data import ReplayBuffer, LazyTensorStorage
@rl_baselines.register("qlearning-discrete")
class QLearningDiscreteSystem(pl.LightningModule, SaveUtils):
    def __init__(
        self,
        cfg: OmegaConf,
        qvalues_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        pl.LightningModule.__init__(self)
        SaveUtils.__init__(self, cfg)
        self.action_spec = environment.action_spec
        self.action_values_module = TensorDictModule(
            qvalues_network,
            in_keys=['observation'], 
            out_keys=['action_values']
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
        self.target_estimator = TensorDictModule(
            QTargetEstimator(self.action_values_module.module, self.cfg.system.gamma),
            in_keys=[
                ('next', 'reward'),
                ('next', 'observation'),
                ('next', 'done')
            ],
            out_keys=['Qtarget']
        )
        self.env = environment
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)
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
        for t in range(max_episode_length):
            with torch.no_grad():
                tensordict = self.policy_explore(obs_dict)
            tensordict = self.env.step(tensordict)
            self.memory_replay.add(tensordict)
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
                optimizer.step()
                self.egreedy_sampler_module.module.step_egreedy()

            cum_rw += reward
            if done.item():
                break

        self.log_dict(
            {
                "Episode Reward": cum_rw
            }, 
            prog_bar=True, 
            on_epoch=True, 
            on_step=False,
            batch_size=1
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
    def from_config(cls, config: Union[str, OmegaConf]) -> QLearningDiscreteSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        type_network = cfg.system.network.type
        env = make_custom_envs(**cfg.system.environment)

        if type_network == "mlp":
            
            network = mlp_builder(
                get_env_obs_dim(env),
                get_env_action_dim(env),
                **cfg.system.network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {type_network} not implemented!")
        
        return cls(cfg, network, env)