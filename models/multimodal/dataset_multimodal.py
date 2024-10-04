import sys
sys.path.append("..")

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from dataset_raster import npz_to_tensor
from dataset_vector import process_HeteroData

from features import feature_order
from models.operators import operator_order

def collate_raster_vector(batch):
    '''Custom collate function to generate batches containing both tensors and HeteroData objects, as DataLoader cannot collate HeteroData objects by default.'''
    # unzip the batch
    raster_samples, vector_samples, operators = zip(*batch)

    # stack / concatenate the raster samples and operators along the batch dimension
    raster_batch = torch.stack(raster_samples, dim=0)
    operators_batch = torch.cat(operators, dim=0)

    # batch the vector samples
    vector_batch = Batch.from_data_list(vector_samples)

    return raster_batch, vector_batch, operators_batch

class BuildingMultimodalDataset(Dataset):
    def __init__(self, 
                 raster_path, 
                 vector_path, 
                 operators, 
                 features, 
                 attach_roads=True, 
                 raster_transform=None, 
                 vector_transform=None, 
                 subset=None):
        '''Stores the directory and filenames of the individual raster (.npz) and vector (.pt) files.'''
        # store the path to the raster and vector files
        self.raster_path = raster_path
        self.vector_path = vector_path

        # get filenames of the individual files, sort the filenames to make them line up
        if not subset:
            self.raster_filenames = sorted([file for file in os.listdir(raster_path) if file.endswith(".npz")])
            self.vector_filenames = sorted([file for file in os.listdir(vector_path) if file.endswith(".pt")])
        # choose only subset if specified
        else:
            self.raster_filenames = sorted([file for file in os.listdir(raster_path) if file.endswith(".npz")])[:subset]
            self.vector_filenames = sorted([file for file in os.listdir(vector_path) if file.endswith(".pt")])[:subset]

        # make sure that the samples line up
        assert len(self.raster_filenames) == len(self.vector_filenames)

        # store indices of the operators within operator_order for slicing in the __getitem__ method
        self.operators = sorted([operator_order.index(operator) for operator in operators if operator in operator_order])
        # store indices of the features within feature_order for slicing in the __getitem__ method
        self.features = sorted([feature_order.index(feature) for feature in features if feature in feature_order])

        # store information on whether roads should be attached
        self.attach_roads = attach_roads

        # store transformations
        self.raster_transform = raster_transform
        self.vector_transform = vector_transform

    def __len__(self):
        '''Enables dataset length calculation.'''
        return len(self.raster_filenames)

    def __getitem__(self, index):
        '''Enables indexing, returns graph and raster representation and generalization operator as label.'''
        # load the raster sample associated with the given index
        raster_filename = self.raster_filenames[index]
        raster_sample_raw = np.load(os.path.join(self.raster_path, raster_filename))

        # convert loaded file to tensor
        raster_sample = npz_to_tensor(raster_sample_raw, attach_roads=self.attach_roads)

        if self.raster_transform:
            raster_sample = self.raster_transform(raster_sample)

        # load the vector sample associated with the given index
        vector_filename = self.vector_filenames[index]
        vector_sample_raw = torch.load(os.path.join(self.vector_path, vector_filename))

        # process the raw HeteroData object according to the information specified in the init method
        vector_sample = process_HeteroData(vector_sample_raw, 
                                           operators=self.operators, 
                                           features=self.features, 
                                           attach_roads=self.attach_roads)

        # extract the operators from the graph object
        operators = vector_sample.y

        if self.vector_transform:
            vector_sample = self.vector_transform(vector_sample)

        return raster_sample, vector_sample, operators