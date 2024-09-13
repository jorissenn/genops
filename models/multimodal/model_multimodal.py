import torch
import torch.nn as nn

from classification_head_multimodal import ClassificationHeadMultimodal

class MultimodalModel(nn.Module):
    def __init__(self, raster_model, vector_model, dummy_raster_sample, dummy_vector_sample, n_classes):
        super(MultimodalModel, self).__init__()
        self.raster_model = raster_model
        self.vector_model = vector_model
        
        # both models are already trained and only require gradient for multimodal classification head
        for param in self.raster_model.parameters():
            param.requires_grad = False
        for param in self.vector_model.parameters():
            param.requires_grad = False

        # remove classification heads
        self.raster_model.classification_heads = nn.Identity()
        self.vector_model.classification_heads = nn.Identity()

        # pass dummy raster and vector samples through the networks to determine the number of output features
        # when the classification heads are missing
        out_raster = self.raster_model(dummy_raster_sample.unsqueeze(0))
        out_vector = self.vector_model(dummy_vector_sample.x_dict, dummy_vector_sample.edge_index_dict)
        n_raster_features = out_raster.shape[1]
        n_vector_features = out_vector.shape[1]

        self.classification_heads = nn.ModuleList([ClassificationHeadMultimodal(n_input_features=n_raster_features + n_vector_features,
                                                                                n_classes=1) for _ in range(n_classes)])

    def forward(self, raster, graph):
        raster_output = self.raster_model(raster)
        vector_output = self.vector_model(graph.x_dict, graph.edge_index_dict)

        # concatenate along feature dimension
        x = torch.cat((raster_output, vector_output), dim=1)

        # apply each classification head and concatenate the results along the final dimension
        outputs = torch.cat([head(x).squeeze(-1).unsqueeze(1) for head in self.classification_heads], dim=1)
        
        return outputs

    def get_n_parameters(self):
        n_parameters = sum(p.numel() for p in self.parameters())
        return n_parameters

    def __str__(self):
        return f'''Multimodal Model with {self.get_n_parameters():,} parameters consisting of
        raster model ({self.raster_model.__class__.__name__}) with {self.raster_model.get_n_parameters():,} parameters and 
        vector model ({self.vector_model.__class__.__name__}) with {self.vector_model.get_n_parameters():,} parameters.'''