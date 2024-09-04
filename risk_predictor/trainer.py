import logging
from tqdm import tqdm

from .data import DataModule

log = logging.getLogger(__name__)

def run(cfg):
    log.info(f"Load datasets")
    data_module = DataModule(cfg)

    train_loader = data_module.train_dataloader()

    best_loss = 9e9
    epoch_resume = 0

    for epoch in range(cfg.trainer.epoch_resume, cfg.trainer.total_epochs):
        train(train_loader, epoch, cfg.trainer.total_epochs)

# def train(model, device, loss_func, optimizer, scheduler, dataloader, current_epoch, total_epochs, print_step=30):
#     model.train()
def train(dataloader, current_epoch, total_epochs, print_step=30):
    progress_bar = tqdm(enumerate(dataloader), desc=f"[{current_epoch:03d}/{total_epochs:03d} epoch train]", total=len(dataloader))

    for idx, batch in progress_bar:
        videos, labels = batch # (batch_size, video_num, target_frame_num, 1, 224, 224), (batch_size, )





