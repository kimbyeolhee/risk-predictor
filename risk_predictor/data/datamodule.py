import logging

import torch

log = logging.getLogger(__name__)

# 임의의 더미 비디오 데이터를 반환
# 한 환자 당 영상 5개, 각 영상의 프레임 수가 각각 (120, 111, 142, 93, 64) 라고 하자.
# 데이터 더미 생성

dummy_video = [
    torch.randn(120, 1, 512, 512), 
    torch.randn(111, 1, 512, 512),  
    torch.randn(142, 1, 512, 512),  
    torch.randn(93, 1, 512, 512),   
    torch.randn(64, 1, 512, 512)    
]

class DataModule:
    def __init__(self):
        pass

    def train_dataloader(self):
        pass
