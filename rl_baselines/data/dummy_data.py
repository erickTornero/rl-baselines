from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import math
from typing import Iterator
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
import rl_baselines

class DummyIterativeRLDataset(IterableDataset):
    def __init__(
        self,
        #total_episodes: int,
    ) -> None:
        pass#self.total_episodes = total_episodes

    def __iter__(self) -> Iterator:
        return iter(range(0, 1))
        #while True:
        #    yield {}
        #worker_info = torch.utils.data.get_worker_info()
        #if worker_info is None:  # single-process data loading, return the full iterator
        #    iter_start = 0
        #    iter_end = self.total_episodes
        #else:  # in a worker process
        #    # split workload
        #    per_worker = int(math.ceil((self.total_episodes) / float(worker_info.num_workers)))
        #    worker_id = worker_info.id
        #    iter_start = worker_id * per_worker
        #    iter_end = min(iter_start + per_worker, self.total_episodes)
        #return iter(range(iter_start, iter_end))
@rl_baselines.register("dummy-rl-datamodule")
class DummyRLDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = DummyIterativeRLDataset()#self.cfg.trainer.max_episodes)
        else:
            raise NotImplementedError("")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None
        )