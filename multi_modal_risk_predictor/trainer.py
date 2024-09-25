import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import DataModule
from .model import MultiModalRiskPredictor
from utils.logging import setup_logger



def train(model, dataloader, current_epoch, total_epochs, device, criterion, optimizer):
    correct = 0
    train_loss_history = []
    size = len(dataloader.dataset)

    progress_bar = tqdm(enumerate(dataloader), desc=f"[{current_epoch:03d}/{total_epochs:03d} epoch train]", total=len(dataloader))

    model.train()
    for idx, batch in progress_bar:
        videos, labels = batch # (batch_size, video_num, target_frame_num, 1, 224, 224), (batch_size, ) # ([2, 5, 10, 1, 224, 224])

        visual_inputs = videos.to(device)
        labels = labels.to(device)

        preds = model(visual_inputs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())
        train_loss_history.append(loss.item())

    correct /= size
    return train_loss_history, correct

def validate(model, dataloader, device, criterion):
    test_loss, correct = 0, 0
    val_loss_history = []
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    progress_bar = tqdm(enumerate(dataloader))

    model.eval()
    with torch.no_grad():
        for idx, batch in progress_bar:
            videos, labels = batch # (batch_size, video_num, target_frame_num, 1, 224, 224), (batch_size, ) # ([2, 5, 10, 1, 224, 224])

            visual_inputs = videos.to(device)
            labels = labels.to(device)

            preds = model(visual_inputs)
            loss = criterion(preds, labels)

            test_loss += loss.item()
            correct += (preds.round() == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

    test_loss /= num_batches
    correct /= size

    return test_loss, correct
            

def run(cfg):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_module = DataModule(cfg)
    logger.info("Load datasets")

    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    logger.info(f"Length of train dataset: {len(train_loader.dataset)}")
    logger.info(f"Length of valid dataset: {len(valid_loader.dataset)}")

    visual_params = cfg.visual_encoder
    model = MultiModalRiskPredictor(visual_params).to(device) # TODO: tabular, text 나중에 추가

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_acc = 0.0

    for epoch in range(cfg.trainer.epoch_resume, cfg.trainer.total_epochs):
        train_loss_history, correct = train(model, train_loader, epoch, cfg.trainer.total_epochs, device, criterion, optimizer)
        val_loss, correct = validate(model, valid_loader, device, criterion)

        logger.info(f"Train Loss: {train_loss_history[-1]}, Val Loss: {val_loss}, Val Acc: {correct}")  
        scheduler.step(val_loss)

        if correct > best_acc:
            best_acc = correct
            logger.info(f"Update new best accuracy: {best_acc}")
            # torch.save(model.state_dict(), f"./checkpoints/{cfg.name}.pth")

    logger.info("Training finished")










