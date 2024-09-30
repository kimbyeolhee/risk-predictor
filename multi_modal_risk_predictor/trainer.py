import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

from .data import DataModule
from .model import MultiModalRiskPredictor, VisualModalRiskPredictor
from utils.logging import setup_logger


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, word_size, model, dataloader, current_epoch, total_epochs, criterion, optimizer):
    correct = 0
    train_loss_history = []
    size = len(dataloader.dataset)
    scaler = GradScaler()
    accumulation_steps = 4  # 그래디언트 누적 횟수

    scaler = GradScaler()

    if rank == 0:
        progress_bar = tqdm(enumerate(dataloader), desc=f"[{current_epoch:03d}/{total_epochs:03d} epoch train]", total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    model.train()
    for idx, batch in progress_bar:
        videos, labels = batch # (batch_size, channel, total_frame_num, 224, 224), (batch_size, ) # torch.Size([1, 1, 1577, 224, 224]), torch.Size([1])
        videos, labels = videos.to(rank), labels.to(rank)

        with autocast(device_type="cuda"):
            preds = model(videos) # (batch_size, )
            loss = criterion(preds, labels.float())
            loss = loss / accumulation_steps # 그래디언트 누적

        scaler.scale(loss).backward()

        if (idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        train_loss_history.append(loss.item() * accumulation_steps)
        predicted = (preds > 0.5).float() # 0.5 기준 이진분류
        correct += (predicted == labels).sum().item()

        if rank == 0:
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps, acc=correct/size * 100)

    acc = correct / size
    return train_loss_history, acc

def validate(rank, model, dataloader, criterion):
    test_loss, correct = 0, 0
    val_loss_history = []
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    if rank == 0:
        progress_bar = tqdm(enumerate(dataloader), desc=f"[{current_epoch:03d}/{total_epochs:03d} epoch validate]", total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    model.eval()
    with torch.no_grad():
        for idx, batch in progress_bar:
            videos, labels = batch # (batch_size, video_num, target_frame_num, 1, 224, 224), (batch_size, ) # ([2, 5, 10, 1, 224, 224])
            videos, labels = videos.to(rank), labels.to(rank)

            with autocast(device_type="cuda"):
                preds = model(videos)
                loss = criterion(preds, labels.float())

            val_loss += loss.item()
            predicted = (preds > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

            if rank == 0:
                progress_bar.set_postfix(loss=loss.item(), acc=correct/size * 100)

    val_loss /= num_batches
    acc = correct / size

    return val_loss, acc
            

def run(rank, world_size, cfg):
    try:
        setup(rank, world_size)

        if rank == 0:
            logger = setup_logger()
            logger.info(f"Running DDP on rank {rank} with world size {world_size}")
        
        torch.manual_seed(cfg.trainer.seed)
        torch.cuda.set_device(rank)

        data_module = DataModule(cfg, rank, world_size)

        if rank == 0:
            logger.info("Load datasets")
            logger.info(f"Length of train dataset: {len(data_module.train_dataloader().dataset)}")
            logger.info(f"Length of valid dataset: {len(data_module.valid_dataloader().dataset)}")

        train_loader = data_module.train_dataloader()
        valid_loader = data_module.valid_dataloader()

        model = VisualModalRiskPredictor(cfg.visual_encoder.model_name_or_path).to(rank)
        model = DDP(model, device_ids=[rank])

        # visual_params = cfg.visual_encoder
        # model = MultiModalRiskPredictor(visual_params).to(device) # TODO: tabular, text 나중에 추가

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        best_acc = 0.0

        for epoch in range(cfg.trainer.epoch_resume, cfg.trainer.total_epochs):
            train_loss_history, train_acc = train(rank, world_size, model, train_loader, epoch, cfg.trainer.total_epochs, criterion, optimizer)
            val_loss, val_acc = validate(rank, model, valid_loader, criterion)

            if rank == 0:
                logger.info(f"Epoch {epoch+1}/{cfg.trainer.total_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                scheduler.step(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                logger.info(f"Update new best accuracy: {best_acc}")
                # torch.save(model.state_dict(), f"./checkpoints/{cfg.name}.pth")

        cleanup()
    finally:
        cleanup()









