import logging
import os

import torch
from torch.utils.data import DataLoader

from .datasets import load_dataset

log = logging.getLogger(__name__)


# Visual DataModule for now.
class DataModule:
    def __init__(self, cfg):
        self.data_cfg = cfg.dataset
        self.dataloader_cfg = cfg.dataloader
        self.train_loader = None
        self.test_loader = None

        # Dataset 생성
        self.train_patient_id_csv_path = self.data_cfg.visual.train.patient_id_csv_path
        self.train_video_csv_path = self.data_cfg.visual.train.video_csv_path
        self.valid_patient_id_csv_path = self.data_cfg.visual.valid.patient_id_csv_path
        self.valid_video_csv_path = self.data_cfg.visual.valid.video_csv_path
        
    def train_dataloader(self):
        dataset = load_dataset("visual", self.train_patient_id_csv_path, self.train_video_csv_path, self.data_cfg)

        self.train_loader = DataLoader(dataset, batch_size=self.dataloader_cfg.visual.batch_size)
        return self.train_loader
    
    def valid_dataloader(self):
        dataset = load_dataset("visual", self.valid_patient_id_csv_path, self.valid_video_csv_path, self.data_cfg)

        self.valid_loader = DataLoader(dataset, batch_size=self.dataloader_cfg.visual.batch_size)
        return self.valid_loader
