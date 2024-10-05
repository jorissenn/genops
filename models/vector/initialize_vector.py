import sys
sys.path.append("..")

import os
import torch

from hgnn import HGNN
from hgt import HGT
from dataset_vector import get_dummy_sample
from features import important_features
from models.operators import elimination_operators, selection_operators

def initialize_vector_model(architecture, sample, **kwargs):
    '''Lazy initializes the specified heterogeneous GNN architecture using the metadata from a single training sample.'''
    assert architecture in ("hgnn", "hgt")

    # extract necessary metadata
    node_types = sample.node_types
    
    node_features = {}
    
    for node_type in node_types:
        node_features[node_type] = sample[node_type]["x"].shape[1]
    
    n_classes = sample["y"].shape[1]
    
    print(f"Number of node features: {node_features}, {n_classes} operators")
    
    metadata = sample.metadata()
    
    # creating model and moving to device
    if architecture == "hgnn":
        model = HGNN(hidden_channels=kwargs["hidden_channels"], 
                     out_channels=n_classes, 
                     num_layers=kwargs["num_layers"], 
                     metadata=metadata, 
                     node_to_predict=kwargs["node_to_predict"])
    elif architecture == "hgt":
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

def load_trained_vector_model(model_filename, vector_path, device):
    '''Loads a trained vector model given a filename.'''
    operator_model_map = {"eli": "elimination", "sel": "selection"}

    # extract necessary information from model filename
    model_filename_split = model_filename.split("_")
    architecture = model_filename_split[0].lower()
    operator_model = operator_model_map[model_filename_split[1]]
    operators_to_predict = elimination_operators if operator_model == "elimination" else selection_operators
    attach_roads = True if model_filename_split[2] == "attachRoadsTrue" else False

    # define important features
    features = important_features[f"{architecture.upper()} {operator_model}"]
    
    # load dummy sample
    dummy_sample_path = os.path.join(vector_path, "training_data", "dummy_sample.pt")
    dummy_sample = get_dummy_sample(dummy_sample_path, 
                                    operators=operators_to_predict, 
                                    features=features, 
                                    attach_roads=attach_roads)

    # initialize the model
    model = initialize_vector_model(architecture=architecture, sample=dummy_sample, hidden_channels=128, num_heads=8, num_layers=3, node_to_predict="focal_building")
    model.to(device)

    # load vector model from checkpoint
    model_path = os.path.join(vector_path, "models", operator_model)
    checkpoint = torch.load(os.path.join(model_path, model_filename), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Vector model successfully loaded.")

    return model