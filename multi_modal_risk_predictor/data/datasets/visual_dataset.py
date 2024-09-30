import pandas as pd
import pydicom
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoLoader:
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self._dcm = None
        self._pixel_array = None
    
    def __len__(self):
        if self._pixel_array is None:
            self._load_dcm()
        return len(self._pixel_array)
    
    def __getitem__(self, idx):
        if self._pixel_array is None:
            self._load_dcm()
        frame = self._pixel_array[idx]
        return self.transform(frame)
    
    def _load_dcm(self):
        self._dcm = pydicom.dcmread(self.video_path)
        self._pixel_array = self._dcm.pixel_array
        



class VisualDataset(Dataset):
    """
    unique patient id를 가진 모든 비디오를 반환
    """
    @staticmethod
    def numpy_to_pil(x):
        return Image.fromarray(x.astype(np.uint8), mode='L')

    def __init__(self, patient_id_csv_path: str, video_csv_path: str, dataset_cfg: dict):
        self.patient_id_csv = pd.read_csv(patient_id_csv_path)
        self.video_path_csv = pd.read_csv(video_csv_path)
        self.dataset_cfg = dataset_cfg

        self.transform = transforms.Compose([
            transforms.Lambda(self.numpy_to_pil), # numpy 배열을 그레이스케일 PIL Image로 변환
            transforms.Resize((self.dataset_cfg.visual.input_size, self.dataset_cfg.visual.input_size)) if self.dataset_cfg.visual.input_size != 512 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # gray scale
        ])

    def __len__(self):
        return len(self.patient_id_csv)
    

    def __getitem__(self, idx):
        patient_id = self.patient_id_csv.iloc[idx]["patient_id"]
        label = self.patient_id_csv.iloc[idx]["label"]
        video_pathes = self.video_path_csv.loc[self.video_path_csv["patient_id"] == patient_id, "video_path"].tolist()

        videos = [VideoLoader(path, self.transform) for path in video_pathes]

        total_frames = sum(len(video) for video in videos)

        combined_video = torch.zeros((1, total_frames, self.dataset_cfg.visual.input_size, self.dataset_cfg.visual.input_size))

        current_frame_idx = 0

        for video in videos:
            for i in range(len(video)):
                combined_video[:, current_frame_idx] = video[i]
                current_frame_idx += 1
        
        label = torch.tensor(label, dtype=torch.float32)

        return combined_video, label 
        
        # for path in video_pathes:
        #     video = self._load_video(path)
        #     videos.append(video)
        
        # videos = torch.stack(videos)
        # label = torch.tensor(label, dtype=torch.float32)
        # return videos, label # (video_num, target_frame_num, 1, 224, 224), (1,)


    # def _load_video(self, video_path: str):
    #     dcm = pydicom.dcmread(video_path)
    #     pixel_array = dcm.pixel_array

    #     num_frames = len(pixel_array)
    #     num_target_frames = self.dataset_cfg.visual.num_target_frames

    #     frames = []
    #     if num_frames >= num_target_frames:
    #         # 균등한 간격으로 프레임 선택
    #         indices = np.linspace(0, num_frames - 1, num_target_frames, dtype=int)

    #         for index in indices:
    #             frame = pixel_array[index]
    #             frame = self.transform(frame)
    #             frames.append(frame)
            
    #     else:
    #         # 모든 프레임을 가져오고 부족한 만큼 패딩
    #         for i in range(num_frames):
    #             frame = pixel_array[i]
    #             frame = self.transform(frame)
    #             frames.append(frame)
            
    #         # 패딩 프레임 추가
    #         num_padding_frames = num_target_frames - num_frames
    #         padding_frame = torch.zeros_like(frames[0])
    #         frames.extend([padding_frame] * num_padding_frames)

    #     return torch.stack(frames)

    
