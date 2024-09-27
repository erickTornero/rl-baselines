import torch
import gymnasium
import numpy as np
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase
from torchrl.data import OneHotDiscreteTensorSpec, CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
def _make_spec(self, td_params):
    pass

def _reset(self, tensordict):
    pass

def _step(tensordict):
    pass

def _set_seed(self, tensordict):
    pass


def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "g": g,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


class CustomCartPoleV0(EnvBase):
    metadata = {

    }
    batch_locked = False

    def __init__(
        self,
        td_params=None,
        seed=None,
        device="cpu"
    ):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self._env = gymnasium.make('CartPole-v0')

        self.action_spec =  OneHotDiscreteTensorSpec(2, shape=(2, ), device=device, dtype=torch.int64,)

        self.full_observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                self._env.observation_space.low, 
                self._env.observation_space.high
            ),
            device=device
        )

        self.state_spec = self.observation_spec.clone()
        self.reward_spec = UnboundedContinuousTensorSpec((1, ), dtype=torch.float32, device=device)

        self.full_done_spec = CompositeSpec(
            done=DiscreteTensorSpec(2, (1, ), device=device, dtype=torch.bool),
            device=device,
        )
        #self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        #self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _set_seed = _set_seed

    def _reset(self, tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(self.batch_size)
        obs, _ = self._env.reset()
        return TensorDict({'observation': obs}, batch_size=batch_size)

    def _step(self, tensordict):
        action = tensordict.get('action')
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(action, np.ndarray):
            if action.ndim > 0:
                if action.ndim == 1:
                    if len(action) == self.action_spec.space.n:
                        action = action.argmax()
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        raise TypeError("action if ndim=1, it must have 1 or len == discrete dim")
                else:
                    raise TypeError("action has more than 1 dim")

        action = np.int64(action)

        obnew, rw, done, _, _     =   self._env.step(action)
        out = TensorDict(
            {
                'observation': obnew,
                'reward': rw,
                'done': done
            }
        )
        return out
