import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class VideoFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VideoFeatureExtractor, self).__init__()

        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1) # attention score를 계산하는 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x : (batch_size * video_num, target_frame_num, c, h, w)
        batch_size, target_frame_num, c, h, w = x.shape
        x = x.view(batch_size * target_frame_num, c, h, w)

        features = self.resnet18(x) # features : (batch_size * target_frame_num, 512)
        features = features.view(batch_size, target_frame_num, -1) 

        lstm_out, _ = self.lstm(features) # lstm_out : (batch_size, target_frame_num, hidden_dim)

        attention_score = self.attention(lstm_out) # attention_score : (batch_size, target_frame_num, 1)
        attention_weight = torch.softmax(attention_score, dim=1) # attention_weight : (batch_size, target_frame_num, 1)
        weighted_features = torch.sum(attention_weight * lstm_out, dim=1) # weighted_features : (batch_size, hidden_dim)

        output = self.fc(weighted_features) # output : (batch_size * video_num, output_dim)

        return output

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.layer_norm1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.layer_norm2(x)
        x = self.fc3(x)
        return x


class VisualEncoder(nn.Module):
    def __init__(self, input_size, num_frames, feature_dim=128, hidden_dim=256, output_dim=128):
        super(VisualEncoder, self).__init__()
        self.feature_extractor = VideoFeatureExtractor(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim)
        self.ffn = FFN(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        # Each patient has multiple videos, and each video has multiple frames.
        # We input each frame into the shared video feature extractor.
        batch_size, video_num, target_frame_num, c, h, w = x.shape # (batch_size, video_num, target_frame_num, 1, 224, 224)

        # Combine batch_size and video_num dimensions
        x = x.view(batch_size * video_num, target_frame_num, c, h, w)
        # Extract features for all videos at once
        video_features = self.feature_extractor(x)  # (batch_size * video_num, output_dim)
        # Reshape back to (batch_size, video_num, output_dim)
        video_features = video_features.view(batch_size, video_num, -1)

        video_features = torch.mean(video_features, dim=1) # (batch_size, output_dim)
        output = self.ffn(video_features) # (batch_size, output_dim)
        return output

class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256):
        super(SpatioTemporalFeatureExtractor, self).__init__()
        self.resnet = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
        self.resnet[0] = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_reduction = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.temporal_attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 프레임 수 절반 줄이기
        x = x[:, ::2, ...]

        # x is a tensor representing a video
        batch_size, frame_num, c, h, w = x.shape # torch.Size([1, 97, 1, 224, 224])

        # ResNet feature extraction
        x = x.view(batch_size * frame_num, c, h, w) # torch.Size([97, 1, 224, 224])
        x = self.resnet(x) # torch.Size([97, 512, 7, 7])

        # Spatial Attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights # torch.Size([97, 512, 7, 7])

        # Feature Reduction
        x = self.feature_reduction(x)

        # temporal attention
        _, c, h, w = x.size()
        x = x.view(batch_size, frame_num, c, h, w)
        x = x.mean([3,4]) # condense the height and width into one dimension # torch.Size([1, 73, 512])

        temporal_weights = self.temporal_attention(x).squeeze(-1)
        temproal_weights = F.softmax(temporal_weights, dim=1) # 정규화를 통한 가중치 계산(프레임의 중요도)
        x = (x * temproal_weights.unsqueeze(-1)).sum(dim=1) # 가중치를 곱한 프레임들의 합 # (batch_size, 512)

        return x


    