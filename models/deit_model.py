import timm
import torch
import torch.nn as nn

class DeiTModel(nn.Module):
    def __init__(self, num_classes):
        super(DeiTModel, self).__init__()
        # Load pre-trained DeiT model
        self.model = timm.create_model('deit_base_patch16_224', pretrained=True)
        # Ganti classifier head dengan jumlah kelas yang sesuai
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
