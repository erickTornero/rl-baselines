import torch
from torchrl.envs import EnvBase

class FiringEndOfLifePolicy:
    def __init__(
        self,
        env: EnvBase,
    ):
        self.env = env
        self.nfiring_actions = 0
        self.fire_index = self.env.unwrapped.get_action_meanings().index('FIRE') if 'FIRE' in self.env.unwrapped.get_action_meanings() else -1

    def __call__(
        self,
        action: torch.Tensor,
        #action_values: torch.Tensor, 
        end_of_life: bool
    ):
        if end_of_life:
            ndim = action.ndim
            l0 = action.shape[0]
            action = self.env.action_spec.encode(self.fire_index)
            if ndim == 2:
                action = action.unsqueeze(0)
                action = action.repeat(l0, 1)
            #action_index = action.argmax(dim=-1, keepdim=True)
            #chosen_q = torch.gather(action_values, -1, action_index)
            self.nfiring_actions += 1
        return action#, chosen_q