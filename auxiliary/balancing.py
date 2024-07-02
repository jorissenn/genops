import numpy as np
import pandas as pd
import random
import math

def assign_labelset(row, labels):
    '''Given a DataFrame row, returns a concatenated string of the specified column names, where the columns = 1.'''
    labelset = []

    for label in labels:
        if label in row and row[label] == 1:
            labelset.append(label.capitalize())

    if labelset:
        return ', '.join(labelset)

    return "None"

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

def lp_resample(buildings, labels, target_size):
    '''Under- and oversamples a given building DataFrame with respect to their labelsets until the DataFrame reaches target size.'''
    # assign labelsets
    buildings = buildings.copy()
    buildings["labelset"] = buildings.apply(lambda row: assign_labelset(row, labels), axis=1)

    # group all buildings by their labelsets and create a dictionary with indices
    labelset_bags = buildings.groupby("labelset").groups
    labelset_bags = {labelset: list(bag) for labelset, bag in labelset_bags.items()}

    # calculate target labelset bag size such that all labelsets are represented equally
    target_labelset_bagsize = math.ceil(target_size / len(labelset_bags))

    # partition the labelset bags into majority and minority labelset bags based on whether they are smaller or larger than the target size
    labelset_bags_minority = [labelset for labelset, bag in labelset_bags.items() if len(bag) < target_labelset_bagsize]
    labelset_bags_majority = [labelset for labelset, bag in labelset_bags.items() if len(bag) >= target_labelset_bagsize]

    # oversample all the minority labelsets until they reach the target labelset bagsize
    for labelset in labelset_bags_minority:
        # get the bag of the current labelset
        cur_bag = labelset_bags[labelset]

        # oversample the bag until it reaches the target labelset bagsize by duplicating elements
        cur_bag_oversampled = (cur_bag * ((target_labelset_bagsize // len(cur_bag)) + 1))[:target_labelset_bagsize]

        # reassign the oversampled bag back to the corresponding labelset
        labelset_bags[labelset] = cur_bag_oversampled

    # undersample all the majority labelsets until they reach the target labelset bagsize
    for labelset in labelset_bags_majority:
        # get the bag of the current labelset
        cur_bag = labelset_bags[labelset]
        
        # undersample the bag until it reaches the target labelset bagsize by randomly choosing elements without replacement
        cur_bag_undersampled = random.sample(cur_bag, k=target_labelset_bagsize)

        # reassign the undersampled bag back to the corresponding labelset
        labelset_bags[labelset] = cur_bag_undersampled

    # extract all indices from the resampled labelset bags dictionary
    resampled_indices = [index for indices in labelset_bags.values() for index in indices]
    
    # reconstruct the resampled buildings DataFrame from the modified labelset bags dictionary    
    buildings_resampled = buildings.iloc[resampled_indices]

    # prune the final DataFrame to match the target size
    buildings_resampled = buildings_resampled.head(target_size).reset_index(drop=True)
    
    return buildings_resampled

