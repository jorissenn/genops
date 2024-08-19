import torch

from hgnn import HGNN
from hgt import HGT

def initialize_gnn(model, sample, **kwargs):
    '''Lazy initializes the specified heterogeneous GNN model using the metadata from a single training sample.'''
    assert model in ("hgnn", "hgt")

    # extract necessary metadata
    node_types = sample.node_types
    
    node_features = {}
    
    for node_type in node_types:
        node_features[node_type] = sample[node_type]["x"].shape[1]
    
    n_classes = sample["y"].shape[1]
    
    print(f"Number of node features: {node_features}, {n_classes} operators")
    
    metadata = sample.metadata()
    
    # creating model and moving to device
    if model == "hgnn":
        model = HGNN(hidden_channels=kwargs["hidden_channels"], 
                     out_channels=n_classes, 
                     num_layers=kwargs["num_layers"], 
                     metadata=metadata, 
                     node_to_predict=kwargs["node_to_predict"])
    elif model == "hgt":
        model = HGT(hidden_channels=kwargs["hidden_channels"], 
                    out_channels=n_classes, 
                    num_heads=kwargs["num_heads"], 
                    num_layers=kwargs["num_layers"], 
                    node_types=node_types, 
                    metadata=metadata, 
                    node_to_predict=kwargs["node_to_predict"])

    # initialize lazy modules
    with torch.no_grad():
        _ = model(sample.x_dict, sample.edge_index_dict)

    return model