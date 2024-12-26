import torch
from torchvision import transforms
from typing import Optional, Tuple
from collections import deque

class Preprocessing:
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

class DQNPreprocessing(Preprocessing):
    def __init__(self, out_size: Tuple[int, int]):
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
        return self.reescale_transform(Y)

class StackObservation:
    def __init__(self, length: int=1):
        self.m = length
        self.reset()

    def reset(self):
        self.queue = deque(maxlen=self.m)

    def push(self, processed: torch.Tensor):
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
    preprocesor = DQNPreprocessing(out_size=(128, 128))
    stack = StackObservation(4)

    #env = GymEnv("Breakout")
    env = GymEnv("SpaceInvaders")
    obs_dict = env.reset()

    last_frame = obs_dict["pixels"]
    processed = preprocesor(last_frame)
    
    stack.push(processed)
    cv2.imshow(f'Reinforce Discrete', processed[0].numpy())
    k = cv2.waitKey(1)
    ts = 0
    done = False
    while not done:
        obs_dict = env.step(TensorDict({'action': env.action_spec.sample()}))
        frame = obs_dict['next', 'pixels']
        done = obs_dict['next', 'done'].item()
        processed = preprocesor(frame, last_frame)

        cv2.imshow(f'Reinforce Discrete', processed[0].numpy())
        k = cv2.waitKey(10)

        stack.push(processed)

        last_frame = frame
        ts += 1
        if done:
            obs_dict = env.reset()
            stack.reset()
    print(f"total timesteps: {ts}")

