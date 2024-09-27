from __future__ import annotations
from typing import Union, List, Optional
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from rl_baselines.common import (
    mlp_builder, 
    make_custom_envs, 
    get_env_obs_dim, 
    get_env_action_dim
)
from rl_baselines.utils.save_utils import SaveUtils
from torchrl.envs import EnvBase
from .action_sampler import CategoricalSampler
from .losses import ReinforceLoss
import pytorch_lightning as pl

class ReinforceDiscreteSystem(pl.LightningModule, SaveUtils):
    def __init__(
        self,
        cfg: OmegaConf,
        policy_network: Union[nn.Sequential, nn.Module],
        environment: EnvBase,
    ) -> None:
        pl.LightningModule.__init__(self)
        SaveUtils.__init__(self, cfg)
        self.policy_module = TensorDictModule(
            policy_network,
            in_keys=['observation'], 
            out_keys=['action_probs']
        )
        self.action_sampler = TensorDictModule(
            CategoricalSampler(),
            in_keys=['action_probs'],
            out_keys=['action']
        )
        self.policy_reinforce = TensorDictSequential(self.policy_module, self.action_sampler)
        self.loss_module = TensorDictModule(
            ReinforceLoss(),
            in_keys=['action_probs', 'action', 'G'],
            out_keys=['pg_loss']
        )
        self.env = environment
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.env = self.env.to(self.device)


    def forward(self, batch) -> TensorDict :
        return self.policy_module(batch)

    def training_step(self, batch, batch_idx):
        # in reinforce a train step we consider a trajectory
        optimizer = self.optimizers()
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
            probs_dict = self.policy_module(tensordict)
            #J = torch.gather(probs_dict['action_probs'], 0, action)
            #J = -torch.log(J) * tensordict['G']
            J = self.loss_module(probs_dict)
            optimizer.zero_grad()
            self.manual_backward(J.get('pg_loss'))
            optimizer.step()
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


    def configure_optimizers(self):
        cfg_optimizer = self.cfg.system.optimizer
        optimizer_class = getattr(optim, cfg_optimizer.name)
        optimizer = optimizer_class(self.policy_module.module.parameters(), **cfg_optimizer.args)
        return optimizer

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
        type_network = cfg.system.network.type
        env = make_custom_envs(cfg.system.environment.name)

        if type_network == "mlp":
            
            network = mlp_builder(
                get_env_obs_dim(env),
                get_env_action_dim(env),
                **cfg.system.network.args,
            )
        else:
            raise NotImplementedError(f"Network Type: {type_network} not implemented!")
        
        return cls(cfg, network, env)