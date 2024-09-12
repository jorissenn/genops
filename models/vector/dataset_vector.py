import os
import torch
from torch_geometric.data import Dataset

class BuildingVectorDataset(Dataset):
    def __init__(self, path, operators, operator_order, features, feature_order, attach_roads=True, transform=None, subset=None):
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
        sample = torch.load(os.path.join(self.path, filename))

        # remove road nodes and edges connecting road nodes to building nodes
        if not self.attach_roads:
            edge_types_to_remove = [edge_type for edge_type in sample.edge_types if "road" in edge_type]
            for edge_type in edge_types_to_remove:
                del sample[edge_type]
            del sample["road"]

        # only return the operators specified in the init method by slicing sample.y and reshaping accordingly
        sample.y = sample.y[self.operators].reshape(1, -1)

        # only return the features specified in the init method by slicing focal and context buildings accordingly
        sample["focal_building"].x = sample["focal_building"].x[:,self.features]
        sample["context_building"].x = sample["context_building"].x[:,self.features]

        # apply given transformation if specified
        if self.transform:
            graph = self.transform(graph)

        return sample