from __future__ import annotations
import torch
from typing import Union
from torchrl.envs import GymEnv, TransformedEnv, EnvBase
from tensordict import TensorDict

class NoopEnvironment(EnvBase):
    def __init__(
        self,
        env: Union[GymEnv, TransformedEnv],
        noop_action_max: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._env = env
        self.noop_action_index = env.unwrapped.get_action_meanings().index('NOOP') if 'NOOP' in env.unwrapped.get_action_meanings() else -1
        self.noop_action_max = noop_action_max

    @property
    def env(self):
        return self._env
    
    def to(self, device: torch.device) -> NoopEnvironment:
        super().to(device)
        self._env = self._env.to(device)
        return self
    
    def render(self):
        return self._env.render()
    
    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _reset(self, tensordict: TensorDict, **kwargs):
        obs = self._env._reset(tensordict, **kwargs)
        if self.noop_action_max > 0 and self.noop_action_index >= 0:
            action_noop = self.action_spec.encode(self.noop_action_index)
            for i in range(self.noop_action_max):
                obs["action"] = action_noop
                obs = self._step(obs)
                obs = obs["next"]
        return obs
    
    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _step(self, tensordict):
        rs = self._env.step(tensordict)
        return rs

    def _set_seed(self, seed):
        return self._env._set_seed(seed)

    @property
    def action_spec(self):
        return self._env.action_spec