# Code from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from classification_head_vector import ClassificationHeadVector

class HGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, metadata, node_to_predict):
        super().__init__()
        self.node_to_predict = node_to_predict

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = pyg_nn.HeteroConv({
                edge_type: pyg_nn.SAGEConv((-1, -1), hidden_channels) for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.classification_heads = nn.ModuleList([ClassificationHeadVector(n_input_features=hidden_channels, n_classes=1) for _ in range(out_channels)])

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        if isinstance(self.classification_heads, nn.ModuleList):
            outputs = torch.cat([head(x_dict[self.node_to_predict]).squeeze(-1).unsqueeze(1) for head in self.classification_heads], dim=1)
            return outputs
        else:
            return x_dict[self.node_to_predict]

    def get_n_parameters(self):
        n_parameters = sum(p.numel() for p in self.parameters())
        return n_parameters

    def __str__(self):
        return f"HGNN with {self.get_n_parameters():,} parameters"