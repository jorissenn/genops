import os
import torch

from cnn import CNN
from vit import ViT

def initialize_raster_model(architecture, operators_to_predict, attach_roads):
    '''Initializes a raster model given an architecture and model.'''
    # determine the number of necessary image channels
    n_channels = 3 if attach_roads else 2
    
    # creating model and moving to device
    if architecture == "cnn":
        model = CNN(n_channels=n_channels, n_classes=len(operators_to_predict))
    elif architecture == "vit":
        model = ViT(channels=n_channels, num_classes=len(operators_to_predict), heads=8)

    return model

def load_trained_raster_model(model_filename, raster_path, device):
    '''Loads a trained raster model given a filename.'''
    operator_model_map = {"eli": "elimination", "sel": "selection"}
    
    # operators for the elimination models
    elimination_operators = ["elimination"]
    # operators for the selection models
    selection_operators = ["aggregation", "typification", "displacement", "enlargement"]
    
    # extract necessary information from model filename
    model_filename_split = model_filename.split("_")
    architecture = model_filename_split[0].lower()
    operator_model = operator_model_map[model_filename_split[1]]
    operators_to_predict = elimination_operators if operator_model == "elimination" else selection_operators
    attach_roads = True if model_filename_split[2] == "attachRoadsTrue" else False

    # initialize the model
    model = initialize_raster_model(architecture, operators_to_predict, attach_roads)

    # load raster model from checkpoint
    model_path = os.path.join(raster_path, "models", operator_model)
    checkpoint = torch.load(os.path.join(model_path, model_filename), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Model successfully loaded.")

    return model