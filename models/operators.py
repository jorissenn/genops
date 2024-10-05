import torch

# operators are always specified in this order
operator_order = ("elimination", "aggregation", "typification", "displacement", "enlargement", "simplification")

# operators for the elimination models
elimination_operators = ["elimination"]

# operators for the selection models
selection_operators = ["aggregation", "typification", "displacement", "enlargement"]

def threshold_dic_to_tensor(threshold_dic):
    '''Converts a dictionary with operators as keys and the respective thresholds to a tensor with the correct
    operator order.'''
    # get thresholds from dictionary
    thresholds = []
    
    for operator in operator_order:
        if operator in threshold_dic:
            thresholds.append(threshold_dic[operator])

    return torch.tensor(thresholds)