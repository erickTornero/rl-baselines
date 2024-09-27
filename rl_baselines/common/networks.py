from typing import Optional, Dict, Union
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ActivationType:
    type: str
    kwargs: Optional[Dict[str, int]]=None


def find_nn_module(name: str, **kwargs) -> nn.Module:
    try:
        module = getattr(nn, name)
    except:
        raise ModuleNotFoundError(f"Module <{name}> not found in torch.nn>")
    return module

def mlp_builder(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_hidden: int=1,
    activation_hidden_info: Union[ActivationType, str]='Tanh',
    activation_out_info: Optional[Union[ActivationType, str]]=None,
) -> nn.Sequential:
    if isinstance(activation_hidden_info, str):
        activation_class = find_nn_module(activation_hidden_info)
        kwargs_activation = {}
    else:
        activation_class = find_nn_module(activation_hidden_info.type)
        kwargs_activation = activation_hidden_info.kwargs if activation_hidden_info.kwargs is not None else {}
    assert n_hidden, "hidden dim must be greater than 0"
    layers = [nn.Linear(in_dim, hidden_dim), activation_class()]
    for _ in range(n_hidden - 1): 
        layers += [
            nn.Linear(hidden_dim, hidden_dim), 
            activation_class(**kwargs_activation) 
        ]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if activation_out_info is not None:
        if isinstance(activation_out_info, str):
            out_fn_class = find_nn_module(activation_out_info)
            kwargs_out = {}
        else:
            out_fn_class = find_nn_module(activation_out_info.type)
            kwargs_out = activation_out_info.kwargs if activation_out_info.kwargs is not None else {}
        layers.append(out_fn_class(**kwargs_out))
    return nn.Sequential(*layers)