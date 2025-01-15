import torch
from typing import Optional
from tensordict import TensorDictBase
from rl_baselines.common.preprocessing import DQNPreprocessing, DQNPreprocessing0
from torchrl.envs import Transform
from collections import deque

class CNNPreprocessing(Transform):
    """
        Compatible with environment transforms of torchrl
        It follows preprocessing instructions from DQN paper Minh et al (2015)
        - max from two last frames
        - compute luminance channel Y
        - reescale to squared image, tipically [84, 84]

    """
    def __init__(
        self,
        in_keys = "pixels",
        out_keys = "preprocessed",
        in_keys_inv = None,
        out_keys_inv = None,
        out_size= (84, 84),
    ):
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.preprocessor = DQNPreprocessing0(out_size)

    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset = self._call(tensordict_reset)
        return super()._reset(tensordict, tensordict_reset)

    def _apply_transform(self, obs: torch.Tensor, last_obs: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.preprocessor(obs, last_obs)

    def _call(
        self,
        tensordict: TensorDictBase,
        last_tensordict: Optional[TensorDictBase]=None,
    ) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform.

        For any operation that relates exclusively to the parent env (e.g. FrameSkip),
        modify the _step method instead. :meth:`~._call` should only be overwritten
        if a modification of the input tensordict is needed.

        :meth:`~._call` will be called by :meth:`TransformedEnv.step` and
        :meth:`TransformedEnv.reset`.

        """
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict.get(in_key, default=None)
            if last_tensordict is not None:
                last_value = last_tensordict.get(in_key, default=None)
            else:
                last_value = None
            if value is not None:
                observation = self._apply_transform(value, last_value)
                tensordict.set(
                    out_key,
                    observation,
                )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {tensordict}"
                )
        return tensordict
    
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """The parent method of a transform during the ``env.step`` execution.

        This method should be overwritten whenever the :meth:`~._step` needs to be
        adapted. Unlike :meth:`~._call`, it is assumed that :meth:`~._step`
        will execute some operation with the parent env or that it requires
        access to the content of the tensordict at time ``t`` and not only
        ``t+1`` (the ``"next"`` entry in the input tensordict).

        :meth:`~._step` will only be called by :meth:`TransformedEnv.step` and
        not by :meth:`TransformedEnv.reset`.

        Args:
            tensordict (TensorDictBase): data at time t
            next_tensordict (TensorDictBase): data at time t+1

        Returns: the data at t+1
        """
        next_tensordict = self._call(next_tensordict, tensordict)
        return next_tensordict

class StackTransform(Transform):
    def __init__(
        self, 
        in_keys = "preprocessed",
        out_keys = "observation",
        in_keys_inv = None,
        out_keys_inv = None,
        length=4
    ):
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.m = length
        self._reset_deque()

    def _reset_deque(self):
        self._queue = deque(maxlen=self.m)

    def _apply_transform(self, obs: torch.Tensor):
        self._push_deque(obs.clone())
        return self._get_stack()

    def _reset(self, tensordict, tensordict_reset):
        self._reset_deque()
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _push_deque(self, processed: torch.Tensor):
        processed = processed.unsqueeze(0)
        if len(self._queue) < self.m:
            for _ in range(self.m - len(self._queue)):
                self._queue.append(torch.zeros_like(processed))
        self._queue.append(processed)

    def _get_stack(self) -> torch.Tensor:
        return torch.concatenate(list(self._queue))
