import sys
sys.path.append("..")

import os
import torch
from torch_geometric.data import Dataset

from features import feature_order
from models.operators import operator_order

def process_HeteroData(sample, operators, features, attach_roads=True):
    '''Processes a raw HeteroData object as loaded from a .pt file by removing road nodes and edges if specified and returning only the specified features and operators.'''
    # remove road nodes and edges connecting road nodes to building nodes
    if not attach_roads:
        edge_types_to_remove = [edge_type for edge_type in sample.edge_types if "road" in edge_type]
        for edge_type in edge_types_to_remove:
            del sample[edge_type]
        del sample["road"]

    # only return the operators specified by slicing sample.y and reshaping accordingly
    sample.y = sample.y[operators].reshape(1, -1)

    # only return the features specified by slicing focal and context buildings accordingly
    sample["focal_building"].x = sample["focal_building"].x[:,features]
    sample["context_building"].x = sample["context_building"].x[:,features]

    return sample

def get_dummy_sample(path, operators, features, attach_roads=True):
    '''Given a path, reads and processes a dummy sample that can be used to initialize the GNNs.'''
    # store indices of the operators within operator_order
    operators = sorted([operator_order.index(operator) for operator in operators if operator in operator_order])
    # store indices of the features within feature_order
    features = sorted([feature_order.index(feature) for feature in features if feature in feature_order])

    # read the dummy sample
    dummy_sample_raw = torch.load(path)

    # process the dummy sample according to the input parameters
    dummy_sample = process_HeteroData(dummy_sample_raw, operators, features, attach_roads)

    return dummy_sample

class BuildingVectorDataset(Dataset):
    def __init__(self, path, operators, features, attach_roads=True, transform=None, subset=None):
        '''Stores the directory and filenames of the individual .pt files.'''
        super().__init__(path, transform)
        # store directory of individual files
        self.path = path
        # get filenames of individual .pt files, choose only subset if specified
        if not subset:
            self.filenames = [file for file in os.listdir(path) if file.endswith(".pt")]
        else:
            self.filenames = [file for file in os.listdir(path) if file.endswith(".pt")][:subset]

        # store indices of the operators within operator_order for slicing in the .get() method
        self.operators = sorted([operator_order.index(operator) for operator in operators if operator in operator_order])
        # store indices of the features within feature_order for slicing in the .get() method
        self.features = sorted([feature_order.index(feature) for feature in features if feature in feature_order])

        # store information on whether roads should be attached
        self.attach_roads = attach_roads

        # store transformation
        self.transform = transform

    def len(self):
        '''Enables dataset length calculation.'''
        return len(self.filenames)

    def get(self, index):
        '''Enables indexing, returns HeteroData object which contains nodes, edges and labels.'''
        # get filename associated with given index
        filename = self.filenames[index]

        # load the file with the filename
        sample_raw = torch.load(os.path.join(self.path, filename))

        # process the raw HeteroData object according to the information specified in the init method
        sample = process_HeteroData(sample_raw, operators=self.operators, features=self.features, attach_roads=self.attach_roads)

        # apply given transformation if specified
        if self.transform:
            graph = self.transform(graph)

        return sample

class BuildingVectorDatasetUUID(Dataset):
    def __init__(self, path, operators, features, uuid, attach_roads=True, transform=None):
        '''Stores the filename of a single .pt file associated with the specified uuid.'''
        super().__init__(path, transform)
        # store directory of individual files
        self.path = path
        # search for the file associated with the provided UUID
        filenames = [file for file in os.listdir(path) if file.endswith(".pt")]
        uuid_index = next(index for index, filename in enumerate(filenames) if uuid in filename)
        self.filenames = [filenames[uuid_index]]

        # store indices of the operators within operator_order for slicing in the .get() method
        self.operators = sorted([operator_order.index(operator) for operator in operators if operator in operator_order])
        # store indices of the features within feature_order for slicing in the .get() method
        self.features = sorted([feature_order.index(feature) for feature in features if feature in feature_order])

        # store information on whether roads should be attached
        self.attach_roads = attach_roads

        # store transformation
        self.transform = transform

    def len(self):
        '''Enables dataset length calculation.'''
        return len(self.filenames)

    def get(self, index):
        '''Enables indexing, returns HeteroData object which contains nodes, edges and labels.'''
        # get filename associated with given index
        filename = self.filenames[index]

        # load the file with the filename
        sample_raw = torch.load(os.path.join(self.path, filename))

        # process the raw HeteroData object according to the information specified in the init method
        sample = process_HeteroData(sample_raw, operators=self.operators, features=self.features, attach_roads=self.attach_roads)

        # apply given transformation if specified
        if self.transform:
            graph = self.transform(graph)

        return sample