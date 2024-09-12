import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BuildingRasterDataset(Dataset):
    def __init__(self, path, operators, attach_roads=True, transform=None, subset=None):
        '''Stores the directory and filenames of the individual .npz files.'''
        # store directory of individual files
        self.path = path
        # get filenames of individual .npz files, choose only subset if specified
        if not subset:
            self.filenames = [file for file in os.listdir(path) if file.endswith(".npz")]
        else:
            self.filenames = [file for file in os.listdir(path) if file.endswith(".npz")][:subset]

        # store indices of the operators within operator_order for slicing in the .__getitem__() method
        self.operators = operators

        # store information on whether roads should be attached
        self.attach_roads = attach_roads

        # store transformation
        self.transform = transform

    def __len__(self):
        '''Enables dataset length calculation.'''
        return len(self.filenames)

    def __getitem__(self, index):
        '''Enables indexing, returns block raster as features and generalization operators as label.'''
        # get filename associated with given index
        filename = self.filenames[index]

        # load the file with the filename
        sample = np.load(os.path.join(self.path, filename))

        # extract the rasters
        focal_building = sample["focal_building"]
        context_buildings = sample["context_buildings"]
        roads = sample["roads"]

        # stack the rasters according to attach_roads
        if self.attach_roads:
            # stack the rasters to shape (3, n_pixels, n_pixels)
            block = np.stack([focal_building, context_buildings, roads], axis=0)
        else:
            # leave out the roads, stack the rasters to shape (2, n_pixels, n_pixels)
            block = np.stack([focal_building, context_buildings], axis=0)

        # convert rasters to tensor
        block = torch.from_numpy(block).float()

        if self.transform:
            block = self.transform(block)

        # collect labels according to specified generalization operators
        operators = [torch.from_numpy(sample[operator]).float() for operator in self.operators]

        # stack the operators to a tensor
        operators = torch.stack(operators, dim=0).float()

        return block, operators