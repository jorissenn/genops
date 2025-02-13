import sys
sys.path.append("../raster")
sys.path.append("../vector")
sys.path.append("..")

import os
import torch

from initialize_raster import initialize_raster_model

from dataset_vector import get_dummy_sample
from initialize_vector import initialize_vector_model
from features import important_features

from dataset_multimodal import BuildingMultimodalDataset
from model_multimodal import MultimodalModel

from models.operators import elimination_operators, selection_operators

def get_constituent_model_filenames(architecture, operator_model, attach_roads):
    '''Given the architecture name of a multimodal model, the model type (elimination / selection) and whether the roads should be attached,
    get the filenames of the previously trained constituent models.'''
    # extract individual architectures
    architecture_raster, architecture_vector = architecture.split("+")[0], architecture.split("+")[1]

    # get the filenames of the trained constituent models
    if operator_model == "elimination":        
        if architecture_raster == "cnn":
            if attach_roads:
                raster_model_filename = "CNN_eli_attachRoadsTrue_4075585p_100000s_40ep_bs512_cuda.pth"
            else:
                raster_model_filename = "CNN_eli_attachRoadsFalse_4067841p_100000s_40ep_bs512_cuda.pth"
        elif architecture_raster == "vit":
            if attach_roads:
                raster_model_filename = "ViT_eli_attachRoadsTrue_20586241p_100000s_100ep_bs512_cuda.pth"
            else:
                raster_model_filename = "ViT_eli_attachRoadsFalse_20059905p_100000s_100ep_bs512_cuda.pth"
    
        if architecture_vector == "hgnn":
            if attach_roads:
                vector_model_filename = "HGNN_eli_attachRoadsTrue_481665p_100000s_80ep_bs512_cuda.pth"
            else:
                vector_model_filename = "HGNN_eli_attachRoadsFalse_215937p_100000s_80ep_bs512_cuda.pth"
        elif architecture_vector == "hgt":
            if attach_roads:
                vector_model_filename = "HGT_eli_attachRoadsTrue_700466p_100000s_70ep_bs512_cuda.pth"
            else:
                vector_model_filename = "HGT_eli_attachRoadsFalse_452687p_100000s_70ep_bs512_cuda.pth"
                
    elif operator_model == "selection":
        if architecture_raster == "cnn":
            if attach_roads:
                raster_model_filename = "CNN_sel_attachRoadsTrue_8893252p_100000s_50ep_bs512_cuda.pth"
            else:
                raster_model_filename = "CNN_sel_attachRoadsFalse_8885508p_100000s_50ep_bs512_cuda.pth"
        elif architecture_raster == "vit":
            if attach_roads:
                raster_model_filename = "ViT_sel_attachRoadsTrue_20783620p_100000s_90ep_bs512_cuda.pth"
            else:
                raster_model_filename = "ViT_sel_attachRoadsFalse_20257284p_100000s_90ep_bs512_cuda.pth"

        if architecture_vector == "hgnn":
            if attach_roads:
                vector_model_filename = "HGNN_sel_attachRoadsTrue_540548p_100000s_80ep_bs512_cuda.pth"
            else:
                vector_model_filename = "HGNN_sel_attachRoadsFalse_271236p_100000s_80ep_bs512_cuda.pth"
        elif architecture_vector == "hgt":
            if attach_roads:
                vector_model_filename = "HGT_sel_attachRoadsTrue_750389p_100000s_130ep_bs512_cuda.pth"
            else:
                vector_model_filename = "HGT_sel_attachRoadsFalse_502610p_100000s_130ep_bs512_cuda.pth"

    return raster_model_filename, vector_model_filename

def load_trained_multimodal_model(model_filename, multimodal_path, raster_path, vector_path, device):
    '''Loads a trained multimodal model given a filename.'''
    operator_model_map = {"eli": "elimination", "sel": "selection"}

    # extract necessary information from model filename
    model_filename_split = model_filename.split("_")
    architecture = model_filename_split[0].lower()
    operator_model = operator_model_map[model_filename_split[1]]
    operators_to_predict = elimination_operators if operator_model == "elimination" else selection_operators
    attach_roads = True if model_filename_split[2] == "attachRoadsTrue" else False

    # extract individual architectures
    architecture_raster, architecture_vector = architecture.split("+")[0].lower(), architecture.split("+")[1].lower()
    assert architecture_raster in ("cnn", "vit")
    assert architecture_vector in ("hgnn", "hgt")

    # define important features
    features = important_features[f"{architecture_vector.upper()} {operator_model}"]

    # load vector dummy sample
    dummy_vector_sample_path = os.path.join(vector_path, "training_data", "dummy_sample.pt")
    dummy_vector_sample = get_dummy_sample(dummy_vector_sample_path, 
                                           operators=operators_to_predict, 
                                           features=features, 
                                           attach_roads=attach_roads)
    dummy_vector_sample = dummy_vector_sample.to(device)

    # load raster dummy sample
    path_to_raster_training_data = os.path.join(raster_path, "training_data", "elimination", "training")
    path_to_vector_training_data = os.path.join(vector_path, "training_data", "elimination", "training")
    dummy_raster_sample = BuildingMultimodalDataset(path_to_raster_training_data, 
                                                    path_to_vector_training_data, 
                                                    operators=operators_to_predict,
                                                    features=features,
                                                    attach_roads=attach_roads,
                                                    raster_transform=None,
                                                    vector_transform=None,
                                                    subset=None)[0][0]
    dummy_raster_sample = dummy_raster_sample.to(device)

    # initialize raster model
    raster_model = initialize_raster_model(architecture_raster, operators_to_predict, attach_roads)

    # initialize vector model
    vector_model = initialize_vector_model(architecture=architecture_vector, sample=dummy_vector_sample, hidden_channels=128, num_heads=8, num_layers=3, node_to_predict="focal_building")

    # initialize the multimodal model
    model = MultimodalModel(raster_model=raster_model, 
                            vector_model=vector_model, 
                            dummy_raster_sample=dummy_raster_sample, 
                            dummy_vector_sample=dummy_vector_sample, 
                            n_classes=len(operators_to_predict))
    model.to(device)

    # load multimodal model from checkpoint
    model_path = os.path.join(multimodal_path, "models", operator_model)
    checkpoint = torch.load(os.path.join(model_path, model_filename), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Multimodal model successfully loaded.")

    return model