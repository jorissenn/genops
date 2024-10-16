import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

from dataset_raster import BuildingRasterDatasetUUID
from thresholds_raster import raster_thresholds

from models.operators import elimination_operators, selection_operators, threshold_dic_to_tensor

def get_activations_raster(model, dataset, batch_size, operators_to_pred, device):
    '''Given a trained raster model, a dataset, a batch size, some operators to predict and a threshold, 
    calculates the activation values on the dataset.'''
    # creating DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # switch model to evaluation mode
    model.eval()

    # storing the activation values
    activations = []
    
    # prediction evaluations should not be part of the computational graph, gradients should not be tracked
    with torch.no_grad():
        for block, operators in dataloader:
            
            # moving the features to device
            block = block.to(device)
            operators = operators.to(device)
    
            # prediction on the trained model results in logits, sigmoid needs to be applied to obtain probabilities
            pred_operators_logits = model(block)
            pred_operators = torch.sigmoid(pred_operators_logits)

            # flatten the activations and convert to list
            activations.extend((pred_operators.flatten().tolist()))

    return activations

def predict_raster_elimination(elimination_model, path_to_raster_data, uuid, attach_roads, device):
    '''Conducts an operator prediction given a raster elimination model and a uuid.'''
    # get architecture name from model object 
    architecture = elimination_model.__class__.__name__

    # extract threshold
    threshold = raster_thresholds[f"{architecture} elimination"]

    # create Dataset and DataLoader
    dataset = BuildingRasterDatasetUUID(path=path_to_raster_data,
                                        operators=elimination_operators,
                                        uuid=uuid,
                                        attach_roads=attach_roads)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # compute prediction through the elimination model
    with torch.no_grad():
        for raster_sample, _ in dataloader:
            raster_sample.to(device)
            pred_elimination_logits = elimination_model(raster_sample)
            pred_elimination = torch.sigmoid(pred_elimination_logits)
            pred_elimination_label = (pred_elimination > threshold).float().squeeze(0)

            return int(pred_elimination_label.item())

def predict_raster_selection(selection_model, path_to_raster_data, uuid, attach_roads, device):
    '''Conducts an operator prediction given a raster selection model and a uuid.'''
    # get architecture name from model object 
    architecture = selection_model.__class__.__name__

    # extract thresholds and convert to tensor
    thresholds = threshold_dic_to_tensor(raster_thresholds[f"{architecture} selection"])

    # create Dataset and DataLoader
    dataset = BuildingRasterDatasetUUID(path=path_to_raster_data,
                                        operators=selection_operators,
                                        uuid=uuid,
                                        attach_roads=attach_roads)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # compute prediction through the selection model
    with torch.no_grad():
        for raster_sample, _ in dataloader:
            raster_sample.to(device)
            pred_selection_logits = selection_model(raster_sample)
            pred_selection = torch.sigmoid(pred_selection_logits)
            pred_selection_label = (pred_selection > thresholds).float().squeeze(0)

    # store the predictions for all operators
    operators_pred = {}

    for i, operator in enumerate(selection_operators):
        operators_pred[operator] = int(pred_selection_label[i].item())

    return operators_pred