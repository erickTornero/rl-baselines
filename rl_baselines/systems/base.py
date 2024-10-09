from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from tensordict import TensorDict
from tqdm import tqdm
from rl_baselines.common import (
    mlp_builder,
    parse_env_cfg,
    init_env_stats
)
from rl_baselines.common.custom_envs import parse_env_cfg
from rl_baselines.utils.save_utils import SaveUtils
from torchrl.envs import EnvBase
import pytorch_lightning as pl
from typing import Union
import cv2
from collections import deque
from typing import Dict

class RLBaseSystem(pl.LightningModule, SaveUtils):
    def __init__(
        self,
        cfg: OmegaConf,
        environment: EnvBase,
    ) -> None:
        pl.LightningModule.__init__(self)
        SaveUtils.__init__(self, cfg)

        self.env = environment
        self.automatic_optimization = False
        self.cum_rewards = deque(maxlen=10)#cfg.checkpoint.average_last_k_episodes)

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)

    def forward(self, batch) -> TensorDict :
        raise NotImplementedError("")
        #pass#return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("")

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
        raise NotImplementedError("")

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        print(f'Loading state dict from {path}')
        # TODO: added strict=False since environment stats is not loaded properly
        # the loader tries to load from env.transform and env._transform, the last one raises missing keys
        messages = self.load_state_dict(ckpt['state_dict'], strict=False)
        # a simple patch to properly load stats
        self.load_env_stats_patch(ckpt)
        if len(messages.missing_keys) > 0:
            print('-'*20 + "\nMissing keys in checkpoint\n-" + '\n-'.join(messages.missing_keys))
            print('Make sure it belongs only to base environments otherwise it may inccur in an error\n')
        if len(messages.unexpected_keys) > 0:
            print('Unexpected keys in checkpoint\n-' + '\n-'.join(messages.unexpected_keys) + '\n' + '-'*20)

    def load_env_stats_patch(self, ckpt):
        env_state_dict = {}
        for name in ckpt['state_dict'].keys():
            if name.find('env.') == 0:
                env_state_dict[name.replace('env.', '')] = ckpt['state_dict'][name]

        if len(env_state_dict) > 0:
            self.env.load_state_dict(env_state_dict)

    def log_dict_with_envname(self, log_dict: Dict[str, Union[torch.Tensor, float]]):
        envname = self.cfg.system.environment.name
        log_dict = {f"{envname}/{k}": v for k, v in log_dict.items()}
        self.log_dict(
            log_dict,
            prog_bar=True, 
            on_epoch=True, 
            on_step=False,
            batch_size=1
        )

    @staticmethod
    def load_env_from_cfg(environment_cfg: DictConfig) -> EnvBase:
        env = parse_env_cfg(**environment_cfg)
        init_env_stats(env)
        return env
    
    @staticmethod
    def load_network_from_cfg(
        network_cfg: OmegaConf, 
        input_dim: int, 
        output_dim: int
    ) -> nn.Module:
        if network_cfg.type == "mlp":
            network = mlp_builder(
                input_dim,
                output_dim,
                **network_cfg.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {network_cfg.type} not implemented!")
        return network

    @classmethod
    def from_config(cls, config: Union[str, OmegaConf]) -> RLBaseSystem:
        raise NotImplementedError("")