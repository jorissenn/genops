import os
import numpy as np
import torch
from torch.utils.data import Dataset

def npz_to_tensor(sample, attach_roads=True):
    '''Converts a raster object as loaded from a .npz file to a tensor by extracting the necessary components
    and stacking them accordingly.'''
    # extract the rasters
    focal_building = sample["focal_building"]
    context_buildings = sample["context_buildings"]
    roads = sample["roads"]

    # stack the rasters according to attach_roads
    if attach_roads:
        # stack the rasters to shape (3, n_pixels, n_pixels)
        block = np.stack([focal_building, context_buildings, roads], axis=0)
    else:
        # leave out the roads, stack the rasters to shape (2, n_pixels, n_pixels)
        block = np.stack([focal_building, context_buildings], axis=0)

    # convert rasters to tensor
    block = torch.from_numpy(block).float()

    return block

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

        # convert loaded file to tensor
        block = npz_to_tensor(sample, attach_roads=self.attach_roads)

        if self.transform:
            block = self.transform(block)

        # collect labels according to specified generalization operators
        operators = [torch.from_numpy(sample[operator]).float() for operator in self.operators]

        # stack the operators to a tensor
        operators = torch.stack(operators, dim=0).float()

        return block, operators

class BuildingRasterDatasetUUID(Dataset):
    def __init__(self, path, operators, uuid, attach_roads=True, transform=None):
        '''Stores the filename of a single .npz file associated with the specified uuid.'''
        # store directory of individual files
        self.path = path
        # search for the file associated with the provided UUID
        filenames = [file for file in os.listdir(path) if file.endswith(".npz")]
        uuid_index = next(index for index, filename in enumerate(filenames) if uuid in filename)
        self.filenames = [filenames[uuid_index]]

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

        # convert loaded file to tensor
        block = npz_to_tensor(sample, attach_roads=self.attach_roads)

        if self.transform:
            block = self.transform(block)

        # collect labels according to specified generalization operators
        operators = [torch.from_numpy(sample[operator]).float() for operator in self.operators]

        # stack the operators to a tensor
        operators = torch.stack(operators, dim=0).float()

        return block, operators