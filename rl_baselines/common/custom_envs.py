import rl_baselines.environments as cenvs
from torchrl.envs import EnvBase
from torchrl.data import OneHotDiscreteTensorSpec, BoundedTensorSpec, TensorSpec
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
    if isinstance(obs_spec, BoundedTensorSpec):
        return obs_spec.shape[0]
    else:
        raise NotImplementedError("")