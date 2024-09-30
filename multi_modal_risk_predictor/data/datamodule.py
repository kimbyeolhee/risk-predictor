import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import load_dataset

log = logging.getLogger(__name__)


# Visual DataModule for now.
class DataModule:
    def __init__(self, cfg, rank, world_size):
        self.data_cfg = cfg.dataset
        self.rank = rank
        self.world_size = world_size

        self.dataloader_cfg = cfg.dataloader
        self.train_loader = None
        self.valid_loader = None

        # Dataset 생성
        self.train_patient_id_csv_path = self.data_cfg.visual.train.patient_id_csv_path
        self.train_video_csv_path = self.data_cfg.visual.train.video_csv_path
        self.valid_patient_id_csv_path = self.data_cfg.visual.valid.patient_id_csv_path
        self.valid_video_csv_path = self.data_cfg.visual.valid.video_csv_path

        self.train_dataset = load_dataset("visual", self.train_patient_id_csv_path, self.train_video_csv_path, self.data_cfg)
        self.valid_dataset = load_dataset("visual", self.valid_patient_id_csv_path, self.valid_video_csv_path, self.data_cfg)
        
    def train_dataloader(self):
        if self.train_loader is None:
            sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.dataloader_cfg.visual.batch_size,
                sampler=sampler,
                num_workers=self.dataloader_cfg.num_workers,
                pin_memory=True
            )
        return self.train_loader
    
    def valid_dataloader(self):
        if self.valid_loader is None:
            sampler = DistributedSampler(self.valid_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.valid_loader = DataLoader(
                self.valid_dataset, 
                batch_size=self.dataloader_cfg.visual.batch_size,
                sampler=sampler,
                num_workers=self.dataloader_cfg.num_workers,
                pin_memory=True
            )
        return self.valid_loader
