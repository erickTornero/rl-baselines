import torch
from typing import Optional, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
import rl_baselines.environments as cenvs
from torchrl.envs import EnvBase, Compose, TransformedEnv, DoubleToFloat, EndOfLifeTransform, Transform
from torchrl.data import OneHotDiscreteTensorSpec, BoundedTensorSpec, TensorSpec
from torch.nn.parameter import UninitializedBuffer
from copy import deepcopy
from rl_baselines.common.preprocessing import DQNPreprocessing, StackObservation
import rl_baselines.common.custom_env_transforms as ctransforms
from torchrl.envs import GymEnv, GymWrapper, EnvBase
from tensordict import TensorDict
import rl_baselines.common.wrap_envs as cwrappers
def make_custom_envs(
    name: str,
    *args,
    **kwargs,
) -> EnvBase:
    try:
        env_class = getattr(cenvs, name)
        return env_class(*args, **kwargs)
    except:
        from torchrl import envs
        if 'render' in kwargs:
            render = kwargs.pop('render')
            if render: kwargs['render_mode'] = 'rgb_array'

        return envs.GymEnv(name, *args, **kwargs)

def parse_env_cfg(
    name: str,
    *args,
    **kwargs,
) -> Union[EnvBase, TransformedEnv]:
    transforms = None
    using_observation = not ('from_pixels' in kwargs) or not kwargs['from_pixels']
    if 'transforms' in kwargs:
        kwargs = deepcopy(kwargs)
        transforms_dict = kwargs.pop('transforms')
        if transforms_dict is not None:
            transforms = parse_transforms_cfg(transforms_dict)
    if 'wrapper' in kwargs:
        kwargs = deepcopy(kwargs)
        wrapper_dict = kwargs.pop('wrapper')
    else:
        wrapper_dict = None

    env = make_custom_envs(name, *args, **kwargs)
    # Automatic adding of DoubleToFloat transform when observation is float64, used by mujoco envs
    if using_observation and 'observation' in env.observation_spec and env.observation_spec['observation'].dtype == torch.float64:
        # add transform DoubleToFloat
        if transforms is not None:
            # has double to float transform already
            if isinstance(transforms, Compose):
                if not any([isinstance(tr, DoubleToFloat) for tr in transforms]):
                    transforms = Compose(DoubleToFloat(in_keys=['observation']), *transforms)
            else:
                if not isinstance(transforms, DoubleToFloat):
                    transforms = Compose(DoubleToFloat(in_keys=['observation']), transforms)
        else:
            transforms = DoubleToFloat(in_keys=['observation'])

    if 'force_fire' in kwargs and kwargs.get('force_fire'):
        eof_transform = EndOfLifeTransform()
        if transforms is None:
            transforms = eof_transform
        else:
            if isinstance(transforms, Compose):
                transforms.append(eof_transform)
            else:
                transforms = Compose(transforms, eof_transform)

    if transforms is not None:
        env = TransformedEnv(env, transforms)

    if wrapper_dict is not None:
        env = wrap_environment(env, wrapper_dict)

    return env

def wrap_environment(
    env: EnvBase,
    wrap_cfg: OmegaConf
) -> EnvBase:
    wrapper_class = getattr(cwrappers, wrap_cfg.type)
    return wrapper_class(env, **wrap_cfg.args)

def get_env_action_dim(
    env: EnvBase
) -> int:
    action_spec = env.action_spec
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        return action_spec.space.n
    elif isinstance(action_spec, TensorSpec):
        if isinstance(action_spec, BoundedTensorSpec):
            return action_spec.shape[0]
        else:
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")

def get_env_obs_dim(
    env: EnvBase
):
    obs_spec = env.observation_spec['observation']
    if isinstance(obs_spec, TensorSpec):
        return obs_spec.shape[0]
    else:
        raise NotImplementedError("")

def parse_transform(
    name: str,
    *args,
    **kwargs,
):
    from torchrl import envs
    try:
        transform_class = getattr(envs, name)
        transform = transform_class(*args, **kwargs)
    except:
        if hasattr(ctransforms, name):
            transform_class = getattr(ctransforms, name)
            transform = transform_class(*args, **kwargs)
        else:
            raise NotImplementedError(f"transform <{name}> not found in torchrl.envs")
    return transform

def parse_transforms_cfg(
    transforms_cfg: OmegaConf,
) -> Optional[Compose]:
    if isinstance(transforms_cfg, DictConfig):
        if len(transforms_cfg) > 0:
            transforms = []
            for name, values in transforms_cfg.items():
                if values is None:
                    values = {}
                transforms.append(parse_transform(name, **values))
            return Compose(*transforms)
        else:
            return None
    elif isinstance(transforms_cfg, ListConfig):
        raise NotImplementedError(f"Not supported list")
        #if len(transforms_cfg) > 0:
        #    for value in transforms_cfg:
        #        pass

def init_env_stats(env) -> None:
    # init env stats if required, for example when using ObservationNorm transform
    if isinstance(env, TransformedEnv):
        # hardcoded init stats
        init_stats_steps = 100
        if isinstance(env.transform, Compose):
            for transform in env.transform:
                if hasattr(transform, 'init_stats'):
                    if isinstance(transform.loc, UninitializedBuffer):
                        transform.init_stats(init_stats_steps)
        else:
            if hasattr(env.transform, 'init_stats'):
                if isinstance(env.transform.loc, UninitializedBuffer):
                    env.transform.init_stats(init_stats_steps)

