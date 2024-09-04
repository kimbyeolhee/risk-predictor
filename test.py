import argparse
import logging
import os

import torch
from omegaconf import OmegaConf

from risk_predictor import run

log = logging.getLogger(__name__)

def main(args):
    cfg = OmegaConf.load(f"./configs/{args.config}.yaml")

    run(cfg)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args , _ = parser.parse_known_args()

    main(args)
