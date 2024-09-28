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
from .action_sampler import CategoricalSampler
from .losses import ReinforceWithBaselineLoss
import pytorch_lightning as pl
import cv2
import rl_baselines

@rl_baselines.register("reinforce-discrete-baseline")
class ReinforceDiscreteBaselineSystem(pl.LightningModule, SaveUtils):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        baseline_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        pl.LightningModule.__init__(self)
        SaveUtils.__init__(self, cfg)
        self.policy_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
        )
        self.baseline_module = TensorDictModule(
            baseline_network,
            in_keys=['observation'], 
            out_keys=['baseline_value']
        )
        self.action_sampler = TensorDictModule(
            CategoricalSampler(),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy_reinforce = TensorDictSequential(self.policy_module, self.action_sampler)
        self.loss_module = TensorDictModule(
            ReinforceWithBaselineLoss(),
            in_keys=['action_probs', 'baseline_value', 'action', 'G'],
            out_keys=['reinforce_loss', 'baseline_loss']
        )
        self.env = environment
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)


    def forward(self, batch) -> TensorDict :
        return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        [optimizer, optimizer_bl] = self.optimizers()
        with torch.no_grad():
            episode_data = self.env.rollout(
                self.cfg.data.max_trajectory_length,
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

        episode_data['G'] = torch.vstack(GL2)

        cum_rw = episode_data['next', 'reward'].sum()
        #import pdb;pdb.set_trace()
        for t_episode in range(len(episode_data) - 1):
            tensordict = episode_data[t_episode]
            #action = tensordict['action']
            # uncomment following line to allow gymenv environment
            #action = tensordict['action'].argmax()
            tensordict = self.policy_module(tensordict)
            tensordict = self.baseline_module(tensordict)
            #J = torch.gather(probs_dict['action_probs'], 0, action)
            #J = -torch.log(J) * tensordict['G']
            losses = self.loss_module(tensordict)
            optimizer.zero_grad()
            self.manual_backward(losses.get('reinforce_loss'))
            optimizer.step()

            optimizer_bl.zero_grad()
            self.manual_backward(losses.get('baseline_loss'))
            optimizer_bl.step()


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
        optimizer = optimizer_class(self.policy_module.module.parameters(), **cfg_optimizer.args)
        optimizer_bl = optimizer_class(self.baseline_module.module.parameters(), **cfg_optimizer.args)
        return [optimizer, optimizer_bl]

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        print(f'Loading state dict from {path}')
        self.load_state_dict(ckpt['state_dict'])

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> ReinforceDiscreteSystem:
        if isinstance(config, str):
            cfg = OmegaConf.load(config)
        else:
            cfg = config

        env = make_custom_envs(**cfg.system.environment)

        if cfg.system.policy_network.type == "mlp":
            
            policy_network = mlp_builder(
                get_env_obs_dim(env),
                get_env_action_dim(env),
                **cfg.system.policy_network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {cfg.system.policy_network.type} not implemented!")

        if cfg.system.baseline_network.type == "mlp":
            
            baseline_network = mlp_builder(
                in_dim=get_env_obs_dim(env),
                out_dim=1,
                **cfg.system.baseline_network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {cfg.system.baseline_network.type} not implemented!")

        return cls(cfg, policy_network, baseline_network, env)