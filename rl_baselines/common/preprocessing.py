import torch
from torchvision import transforms
from typing import Optional, Tuple, Union
from collections import deque
from torch import nn
import torchvision.transforms.functional as F

class Preprocessing:
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

class DQNPreprocessing0(Preprocessing):
    def __init__(self, out_size):
        from torchvision import transforms
        self.reescale_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(out_size),
            transforms.PILToTensor()
        ])

    def get_luminance_channel(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        assert rgb_tensor.ndim == 3, "dim of input must be 3 [W, H, C]"
        Y = 0.2126 * rgb_tensor[:, :, 0] + 0.7152 * rgb_tensor[:, :, 1] + 0.0722 * rgb_tensor[:, :, 2]
        Y = Y.to(dtype=rgb_tensor.dtype)
        return Y

    def __call__(
        self, 
        rgb_tensor: torch.Tensor, 
        last_frame: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        if last_frame is not None:
            tns = torch.maximum(rgb_tensor, last_frame)
        else:
            tns = rgb_tensor
        Y = self.get_luminance_channel(tns)
        return self.reescale_transform(Y).squeeze(0)


class DQNPreprocessing(Preprocessing):
    def __init__(
        self,
        out_size: Tuple[int, int],
        interpolation: Union[str, F.InterpolationMode]="BILINEAR"
    ):
        super().__init__()
        self.max_tr = MaxTransform()
        self.transform = transforms.Compose([
            LuminanceTransform(),
            ReescaleTransform(out_size, interpolation)
        ])
    def __call__(
        self,
        rgb_tensor: torch.Tensor,
        last_frame: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        out = self.max_tr(rgb_tensor, last_frame)
        return self.transform(out)

class ReescaleTransform(nn.Module):
    def __init__(
        self,
        out_size: Tuple[int, int],
        interpolation: Union[str, F.InterpolationMode]="BILINEAR",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.out_size = out_size
        if isinstance(interpolation, F.InterpolationMode):
            self.interpolation = interpolation
        elif isinstance(interpolation, str):
            self.interpolation = F.InterpolationMode[interpolation.upper()]
        else:
            raise NotImplementedError("")

    def __call__(self, gray_image: torch.Tensor) -> torch.Tensor:
        if gray_image.ndim == 2:
            return F.resize(gray_image.unsqueeze(0), self.out_size, self.interpolation).squeeze(0)
        else:
            return F.resize(gray_image, self.out_size, self.interpolation)

class LuminanceTransform(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, rgb_tensor) -> torch.Tensor:
        assert rgb_tensor.ndim == 3, "dim of input must be 3 [W, H, C]"
        Y = 0.2126 * rgb_tensor[:, :, 0] + 0.7152 * rgb_tensor[:, :, 1] + 0.0722 * rgb_tensor[:, :, 2]
        #Y = 0.299 * rgb_tensor[:, :, 0] + 0.587 * rgb_tensor[:, :, 1] + 0.114 * rgb_tensor[:, :, 2]
        Y = Y.to(dtype=rgb_tensor.dtype)
        return Y

class MaxTransform(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        rgb_tensor: torch.Tensor,
        last_frame: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        if last_frame is not None:
            tns = torch.maximum(rgb_tensor, last_frame)
        else:
            tns = rgb_tensor
        return tns


class StackObservation:
    def __init__(self, length: int=1):
        self.m = length
        self.reset()

    def reset(self):
        self.queue = deque(maxlen=self.m)

    def push(self, processed: torch.Tensor):
        #processed = processed.unsqueeze(0)
        if len(self.queue) < self.m:
            for _ in range(self.m - len(self.queue)):
                self.queue.append(torch.zeros_like(processed))
        self.queue.append(processed)

    def get(self) -> torch.Tensor:
        return torch.concatenate(list(self.queue))

    def __call__(self, processed: torch.Tensor) -> torch.Tensor:
        pass

if __name__ == "__main__":
    from torchrl.envs import GymEnv
    from tensordict import TensorDict
    import cv2
    #preprocesor = DQNPreprocessing(out_size=(128, 128))
    preprocesor2 = DQNPreprocessing(out_size=(128, 128), interpolation="bilinear")
    #F.InterpolationMode.LANCZOS
    # best bicubic
    stack = StackObservation(4)

    env = GymEnv("Breakout")
    #env = GymEnv("SpaceInvaders")
    obs_dict = env.reset()

    cv2.imshow(f'Reinforce Discrete', processed[0].numpy())
    k = cv2.waitKey(1)
    ts = 0
    done = False
    cum_reward = 0
    L = []
    while not done:
        obs_dict = env.step(TensorDict({'action': env.action_spec.sample()}))
        frame = obs_dict['next', 'pixels']
        done = obs_dict['next', 'done'].item()
        reward = obs_dict['next', 'reward'].item()
        L.append(reward)
        if ts % 20 == 0:
            #print(L)
            L = []
        cum_reward += reward
        processed = preprocesor(frame, last_frame)
        processed2 = preprocesor2(frame, last_frame)
        out = torch.concatenate((processed[0], processed2), dim=1)
        #import pdb;pdb.set_trace()

        #print(reward)
        print(processed.sum().item(), processed2.sum().item())


        cv2.imshow(f'Reinforce Discrete', out.numpy())
        k = cv2.waitKey(30)

        stack.push(processed)

        last_frame = frame
        ts += 1
        if done:
            obs_dict = env.reset()
            stack.reset()
    #print(L)
    print(f"total timesteps: {ts}, cum reward -> {cum_reward}")

