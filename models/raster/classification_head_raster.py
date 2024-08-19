import torch.nn as nn

class ClassificationHeadRaster(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super(ClassificationHeadRaster, self).__init__()
        
        # the number of input features of first fully-connected layer are calculated by multiplying number of output channels of last 
        # convolutional layer by (image size after all pooling operations)^2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_input_features, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=n_classes)
        )

    def forward(self, x):
        return self.fc(x)