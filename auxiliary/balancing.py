import numpy as np
import pandas as pd
import random
import math

def calculate_imbalance_measures(buildings, operators):
    '''Given a DataFrame with buildings and a list of generalization operators, calculates the mean imbalance ratio and the coefficient of variation.'''
    # extraction of all labels as numpy array
    buildings_genops = buildings[list(operators)].to_numpy()
    
    # determine counts of labels
    label_counts = np.sum(buildings_genops, axis=0)
    
    # calculate imbalance ratio per label
    ir_per_label = np.max(label_counts) / label_counts
    
    # calculate mean imbalance ratio
    mean_ir = np.mean(ir_per_label)
    
    # calculate coefficient of variation of imbalance ratio per label
    ir_per_label_sigma = np.sqrt(np.sum(((ir_per_label - mean_ir)**2)/(ir_per_label.shape[0] - 1)))
    cvir = ir_per_label_sigma / mean_ir

    return mean_ir, cvir

def lp_resampling(buildings, target_size):
    '''Under- and oversamples a given building DataFrame with respect to their labelsets until the DataFrame reaches target size.'''
    # group all buildings by their labelsets and create a dictionary with indices
    labelset_bags = buildings.groupby("labelset").groups
    labelset_bags = {labelset: list(bag) for labelset, bag in labelset_bags.items()}

    # calculate target labelset bag size such that all labelsets are represented equally
    target_labelset_bagsize = math.ceil(target_size / len(labelset_bags))

    # oversample or undersample all the labelsets until they reach the target labelset bagsize
    for labelset in labelset_bags:
        # get the bag of the current labelset
        cur_bag = labelset_bags[labelset]

        # resample the bag until it reaches the target labelset bagsize
        cur_bag_resampled = random.choices(cur_bag, k=target_labelset_bagsize)

        # reassign the resampled bag back to the corresponding labelset
        labelset_bags[labelset] = cur_bag_resampled

    # extract all indices from the resampled labelset bags dictionary
    resampled_indices = [index for indices in labelset_bags.values() for index in indices]
    
    # reconstruct the resampled buildings DataFrame from the modified labelset bags dictionary    
    buildings_resampled = buildings.iloc[resampled_indices]

    # prune the final DataFrame to match the target size
    buildings_resampled = buildings_resampled.head(target_size).reset_index(drop=True)
    
    return buildings_resampled

