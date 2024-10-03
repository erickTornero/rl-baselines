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
from .action_sampler import ContinuousSampler
from .losses import ReinforceContinuousLoss
import pytorch_lightning as pl
import cv2
import rl_baselines
from torchrl.data import BoundedTensorSpec, ReplayBuffer, LazyTensorStorage
from collections import deque

@rl_baselines.register("reinforce-continuous")
class ReinforceContinuousSystem(pl.LightningModule, SaveUtils):
    def __init__(
        self,
        cfg: OmegaConf,
        mean_network: Union[nn.Sequential, nn.Module],
        std_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        pl.LightningModule.__init__(self)
        SaveUtils.__init__(self, cfg)
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

        self.action_sampler = TensorDictModule(
            ContinuousSampler(environment.action_spec),
            in_keys=['mean_action', 'std_action'],
            out_keys=['action']
        )
        self.policy_reinforce = TensorDictSequential(
            self.mean_module,
            self.std_module,
            self.action_sampler
        )
        self.loss_module = TensorDictModule(
            ReinforceContinuousLoss(self.cfg.system.gamma),
            in_keys=['mean_action', 'std_action', 'action', 'G', 't'],
            out_keys=['pg_loss']
        )
        self.env = environment
        self.automatic_optimization = False
        self.cum_rewards = deque(maxlen=10)#cfg.checkpoint.average_last_k_episodes)

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)


    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        optimizer = self.optimizers()
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

        returns = torch.vstack(GL2)
        norm_returns = (returns - returns.mean())/(returns.std() + 1e-8)
        episode_data['G'] = norm_returns
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
            #probs_dict = self.policy_reinforce(tensordict)
            loss_dict = self.loss_module(probs_dict)
            optimizer.zero_grad()
            self.manual_backward(loss_dict.get('pg_loss').mean())
            optimizer.step()
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
        self.log_dict(
            {
                "Episode Reward": cum_rw,
                "Avg Episode Reward": avg_rw,
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
                action_dict = self.policy_reinforce(obs_dict)
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
        optimizer = optimizer_class(
            list(self.mean_module.parameters()) + list(self.std_module.parameters()), 
            **cfg_optimizer.args
        )
        return optimizer

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        print(f'Loading state dict from {path}')
        self.load_state_dict(ckpt['state_dict'])

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceContinuousSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config
        env = make_custom_envs(**cfg.system.environment)

        if cfg.system.mean_network.type == "mlp":
            
            mean_network = mlp_builder(
                get_env_obs_dim(env),
                get_env_action_dim(env),
                **cfg.system.mean_network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {cfg.system.mean_network.type} not implemented!")
    
        if cfg.system.std_network.type == "mlp":
            
            std_network = mlp_builder(
                get_env_obs_dim(env),
                get_env_action_dim(env),
                **cfg.system.std_network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {cfg.system.std_network.type} not implemented!")
    


        
        return cls(cfg, mean_network, std_network, env)