import torch.nn as nn

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
