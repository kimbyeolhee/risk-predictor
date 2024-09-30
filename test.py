import argparse
import logging
import os
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp

from multi_modal_risk_predictor import run

log = logging.getLogger(__name__)

def main(args):
    cfg = OmegaConf.load(f"./configs/{args.config}.yaml")
    world_size = torch.cuda.device_count()

    try:
        mp.spawn(run, args=(world_size, cfg), nprocs=world_size, join=True)
    finally:
        # 모든 자식 프로세스 종료
        for p in mp.active_children():
            p.terminate()
        # 모든 자식 프로세스 종료 후 모든 메모리 해제
        torch.cuda.empty_cache()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args , _ = parser.parse_known_args()
    
    main(args)
