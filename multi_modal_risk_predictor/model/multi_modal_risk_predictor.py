import torch.nn as nn
from torchvision.models.video import mc3_18, r3d_18

from .module import VisualEncoder
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
    def __init__(self, model_name, pretrained=True):
        super(VisualModalRiskPredictor, self).__init__()
        self.model_name = model_name

        if model_name == "mc3_18":
            self.base_model = mc3_18(pretrained=pretrained)
        elif model_name == "r3d_18":
            self.base_model = r3d_18(pretrained=pretrained)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        self.base_model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False) # GrayScale로 변경
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.base_model(x).squeeze(-1) # (batch_size, )