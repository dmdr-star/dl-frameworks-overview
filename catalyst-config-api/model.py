import torch.nn as nn

from torchvision.models import resnet18


class MyModel(nn.Module):
    def __init__(self, num_classes=1):
        # super().__init__()
        self.backbone = resnet18(num_classes=num_classes)

    def forward(self, batch):
        return self.backbone(batch)
