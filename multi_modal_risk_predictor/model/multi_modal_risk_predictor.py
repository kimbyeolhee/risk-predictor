import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mc3_18, r3d_18

from .module import VisualEncoder, SpatioTemporalFeatureExtractor

class MultiModalRiskPredictor(nn.Module):
    def __init__(self, visual_params):
        super(MultiModalRiskPredictor, self).__init__()
        self.visual_encoder = VisualEncoder(**visual_params)

        self.classifier = nn.Linear(visual_params['output_dim'], 1) # 지금은 우선 fusion 없이 진행

    def forward(self, visual_inputs):
        visual_features = self.visual_encoder(visual_inputs)

        output = self.classifier(visual_features)
        return output.squeeze(-1)


class VisualModalRiskPredictor(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128, output_dim=1):
        super(VisualModalRiskPredictor, self).__init__()
        self.feature_extractor = SpatioTemporalFeatureExtractor(hidden_dim=feature_dim)
        self.attention = nn.Linear(feature_dim, 1)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x is a list of tensors, each tensor represents a video
        video_features = []
        for video in x:
            features = self.feature_extractor(video)
            video_features.append(features)
        
        video_features = torch.stack(video_features, dim=1) # (batch_size, num_videos, feature_dim)

        attention_scores = self.attention(video_features).squeeze(-1) # (batch_size, num_videos, 1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1) # (batch_size, num_videos, 1)
        weighted_features = torch.sum(attention_weights * video_features, dim=1) # (batch_size, feature_dim)

        # classification
        x = F.relu(self.fc1(weighted_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)