import torch
import torch.nn as nn
import torch.nn.functional as F

from classification_head_raster import ClassificationHeadRaster

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # appending a classification head for each class to be predicted
        self.classification_heads = nn.ModuleList([ClassificationHeadRaster(n_input_features=128*32*32, n_classes=1) for _ in range(n_classes)])

    def forward(self, x):
        # input dimension = 256
        x = self.pool(F.relu(self.conv1(x)))
        # input dimension = 128
        x = self.pool(F.relu(self.conv2(x)))
        # input dimension = 64
        x = self.pool(F.relu(self.conv3(x)))
        # input dimension = 32

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