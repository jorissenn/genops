# Code from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

from classification_head_vector import ClassificationHeadVector

class HGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, node_to_predict):
        super().__init__()
        self.node_to_predict = node_to_predict

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = pyg_nn.Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = pyg_nn.HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.classification_heads = nn.ModuleList([ClassificationHeadVector(n_input_features=hidden_channels, n_classes=1) for _ in range(out_channels)])

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        if isinstance(self.classification_heads, nn.ModuleList):
            outputs = torch.cat([head(x_dict[self.node_to_predict]).squeeze(-1).unsqueeze(1) for head in self.classification_heads], dim=1)
            return outputs
        else:
            return x_dict[self.node_to_predict]

    def get_n_parameters(self):
        n_parameters = sum(p.numel() for p in self.parameters())
        return n_parameters

    def __str__(self):
        return f"HGT with {self.get_n_parameters():,} parameters"