# Code from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from classification_head_raster import ClassificationHeadRaster

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # appending a classification head for each class to be predicted
        self.classification_heads = nn.ModuleList([ClassificationHeadRaster(n_input_features=256*7*7, n_classes=1) for _ in range(n_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if isinstance(self.classification_heads, nn.ModuleList):
            # apply each classification head and concatenate the results along the final dimension
            outputs = torch.cat([head(x).squeeze(-1).unsqueeze(1) for head in self.classification_heads], dim=1)
            return outputs
        else:
            return x

    def get_n_parameters(self):
        n_parameters = sum(p.numel() for p in self.parameters())
        return n_parameters

    def __str__(self):
        return f"CNN with {self.get_n_parameters():,} parameters"
        