import torch.nn as nn

class ClassificationHeadMultimodal(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super(ClassificationHeadMultimodal, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=n_input_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=n_classes)
        )

    def forward(self, x):
        return self.fc(x)