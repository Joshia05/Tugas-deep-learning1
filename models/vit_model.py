# models/vit_model.py
import torch
import torch.nn as nn
from timm import create_model

class ViTModel(nn.Module):
    def __init__(self, num_classes=5):  # Ganti 10 menjadi 5
        super(ViTModel, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

