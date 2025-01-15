import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def reshape_stack(data: torch.Tensor) -> torch.Tensor:
    if data.ndim ==3:
        data = torch.hstack([data[i] for i in range(data.shape[-3])])
    elif data.ndim == 4:
        hdata = torch.concatenate([data[:, i] for i in range(data.shape[-3])], dim=-1)
        vdata = torch.concatenate([hdata[i] for i in range(hdata.shape[0])], dim=0)
        data = vdata
    return data

def plot_stack_batch(data: torch.Tensor):
    data = reshape_stack(data)
    data_np = data.cpu().numpy()
    plt.imshow(data_np, )#cmap='gray', vmin=0, vmax=255)
    plt.show()

def plot_rgb_stack(
    frame: torch.Tensor, 
    stack: torch.Tensor, 
    next_stack: torch.Tensor
):
    rz_stack = F.resize(stack, (frame.shape[0], frame.shape[0]))
    nxz_stack = F.resize(next_stack, (frame.shape[0], frame.shape[0]))
    rst = reshape_stack(rz_stack).unsqueeze(-1).repeat(1, 1, 3)
    rnst = reshape_stack(nxz_stack).unsqueeze(-1).repeat(1, 1, 3)
    data = torch.concat((frame, rst, rnst), dim=1)
    
    return data


def check_replay_buffer(
    replay_buffer,
    key1,
    key2,
    batch_size: int,
    done_offset: int=0
):
    sz = len(replay_buffer)
    dones_trues = replay_buffer['next', 'done'].squeeze(-1).nonzero().squeeze(-1).cpu().tolist()
    index_done = 0
    total_diff = 0.0
    it = done_offset
    #import pdb;pdb.set_trace()
    while it < sz:
        try:
            upto = dones_trues[index_done]
        except:
            # in case final item is not done=true
            upto = sz
        nbatches = (upto - it + 1)//batch_size
        nbatches = nbatches if (upto - it + 1) % batch_size == 0 else (nbatches + 1)
        start_index = it
        end_index = min(it + batch_size, upto + 1)
        for _ in range(nbatches):
            data1 = replay_buffer[key1][start_index:end_index]
            data2 = replay_buffer[key2][start_index:end_index]
            # compute difference
            diff = ((data1 - data2)**2).sum().item()
            total_diff += diff

            start_index += batch_size
            end_index += batch_size
            end_index = min(end_index, upto + 1)
        index_done += 1
        it = upto + 1 + done_offset
    return total_diff