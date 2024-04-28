import torch

def calculate_conf_matrix(true_labels, pred_labels):
    '''Given two tensors true_labels and pred_labels containing 0s and 1s, calculates the confusion matrix.'''
    tp = torch.sum((pred_labels == 1) & (true_labels == 1)).float().item() # true positives
    fp = torch.sum((pred_labels == 1) & (true_labels == 0)).float().item() # false positives
    tn = torch.sum((pred_labels == 0) & (true_labels == 0)).float().item() # true negatives
    fn = torch.sum((pred_labels == 0) & (true_labels == 1)).float().item() # false negatives

    return tp, fp, tn, fn

def calculate_metrics(tp, fp, tn, fn):
    '''Given number of true positives, false positives, true negatives and false negatives, 
    calculates accuracy, precision, recall and F1-score.'''
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score