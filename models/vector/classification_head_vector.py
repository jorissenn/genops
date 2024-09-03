import torch.nn as nn
import torch_geometric.nn as pyg_nn

class ClassificationHeadVector(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            pyg_nn.Linear(in_channels=n_input_features, out_channels=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            pyg_nn.Linear(in_channels=128, out_channels=n_classes),
        )

    def forward(self, x):
        return self.fc(x)