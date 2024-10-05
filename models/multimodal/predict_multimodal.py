import sys
sys.path.append("..")
sys.path.append("../vector")

import torch
from torch.utils.data import DataLoader

from features import important_features

from dataset_multimodal import BuildingMultimodalDatasetUUID, collate_raster_vector

from models.operators import elimination_operators, selection_operators, threshold_dic_to_tensor

def get_activations_multimodal(model, dataset, batch_size, operators_to_pred, device):
    '''Given a trained multimodal model, a dataset, a batch size, some operators to predict and a threshold, 
    calculates the activation values on the dataset.'''
    # creating DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raster_vector)

    # switch model to evaluation mode
    model.eval()

    # storing the activation values
    activations = []
    
    # prediction evaluations should not be part of the computational graph, gradients should not be tracked
    with torch.no_grad():
        for raster, vector, operators in dataloader:
            # moving the features to device
            raster = raster.to(device)
            vector = vector.to(device)
            operators = operators.to(device)
    
            # prediction on the trained model results in logits, sigmoid needs to be applied to obtain probabilities
            pred_operators_logits = model(raster, vector)
            pred_operators = torch.sigmoid(pred_operators_logits)

            # flatten the activations and convert to list
            activations.extend((pred_operators.flatten().tolist()))

    return activations

def predict_multimodal_elimination(elimination_model, path_to_raster_data, path_to_vector_data, uuid, attach_roads, threshold, device):
    '''Conducts an operator prediction given a multimodal elimination model and a uuid.'''
    # define important features
    architecture = elimination_model.vector_model.__class__.__name__
    features = important_features[f"{architecture} elimination"]

    # create Dataset and DataLoader
    dataset = BuildingMultimodalDatasetUUID(raster_path=path_to_raster_data,
                                            vector_path=path_to_vector_data,
                                            operators=elimination_operators,
                                            features=features,
                                            uuid=uuid,
                                            attach_roads=attach_roads)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_raster_vector)

    # compute prediction through the elimination model
    with torch.no_grad():
        for raster_sample, vector_sample, _ in dataloader:
            raster_sample.to(device)
            vector_sample.to(device)
            pred_elimination_logits = elimination_model(raster_sample, vector_sample)
            pred_elimination = torch.sigmoid(pred_elimination_logits)
            pred_elimination_label = (pred_elimination > threshold).float().squeeze(0)

            return int(pred_elimination_label.item())

def predict_multimodal_selection(selection_model, path_to_raster_data, path_to_vector_data, uuid, attach_roads, thresholds, device):
    '''Conducts an operator prediction given a multimodal selection model and a uuid.'''
    # define important features
    architecture = selection_model.vector_model.__class__.__name__
    features = important_features[f"{architecture} selection"]
    
    # create Dataset and DataLoader
    dataset = BuildingMultimodalDatasetUUID(raster_path=path_to_raster_data,
                                            vector_path=path_to_vector_data,
                                            operators=selection_operators,
                                            features=features,
                                            uuid=uuid,
                                            attach_roads=attach_roads)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_raster_vector)

    # convert thresholds to tensor
    thresholds = threshold_dic_to_tensor(thresholds)

    # compute prediction through the selection model
    with torch.no_grad():
        for raster_sample, vector_sample, _ in dataloader:
            raster_sample.to(device)
            vector_sample.to(device)
            pred_selection_logits = selection_model(raster_sample, vector_sample)
            pred_selection = torch.sigmoid(pred_selection_logits)
            pred_selection_label = (pred_selection > thresholds).float().squeeze(0)

    # store the predictions for all operators
    operators_pred = {}

    for i, operator in enumerate(selection_operators):
        operators_pred[operator] = int(pred_selection_label[i].item())

    return operators_pred