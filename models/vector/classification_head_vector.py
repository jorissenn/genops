import torch.nn as nn
import torch_geometric.nn as pyg_nn

class ClassificationHeadVector(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            pyg_nn.Linear(n_input_features, n_input_features//2),
            nn.ReLU(inplace=True),
            pyg_nn.Linear(n_input_features//2, n_classes)
        )

    def forward(self, x):
        return self.fc(x)