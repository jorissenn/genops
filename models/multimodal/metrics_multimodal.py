import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from dataset_multimodal import collate_raster_vector

def get_metrics_multimodal(model, dataset, batch_size, operators_to_pred, threshold, device):
    '''Given a trained multimodal model, a dataset, a batch size, some operators to predict and a threshold, 
    calculates and returns accuracy metrics and confusion matrix.'''
    # creating DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raster_vector)

    # switch model to evaluation mode
    model.eval()

    # storing the predictions and true labels from every batch for every operator
    true_operators_all = {}
    pred_operators_all = {}
    
    for operator_name in operators_to_pred:
        true_operators_all[operator_name] = []
        pred_operators_all[operator_name] = []
    
    # prediction evaluations should not be part of the computational graph, gradients should not be tracked
    with torch.no_grad():
        for raster, vector, operators in dataloader:
            # moving the features to device
            raster = raster.to(device)
            vector = vector.to(device)
            operators = operators.to(device)
    
            # prediction on the trained model results in logits, sigmoid needs to be applied to obtain probabilities
            pred_operators_logits = model(raster, vector)
            pred_operators = torch.sigmoid(pred_operators_logits)
            pred_operators_labels = (pred_operators > threshold).float()  # thresholding
    
            # storing true labels and predictions for every operator
            for i, operator_name in enumerate(operators_to_pred):
                # extracting true and predicted operator
                true_operator_batch = operators[:,i]
                pred_operator_batch = pred_operators_labels[:,i]
    
                # collect data for metrics calculation
                true_operators_all[operator_name].append(true_operator_batch.cpu())
                pred_operators_all[operator_name].append(pred_operator_batch.cpu())
    
    metrics = {"operator": [], 
               "conf_matrix": [],
               "accuracy": [], 
               "precision": [], 
               "recall": [], 
               "f1_score": []}
    
    for operator_name in operators_to_pred:
        # convert lists to tensors
        cur_true_operator = torch.cat(true_operators_all[operator_name])
        cur_pred_operator = torch.cat(pred_operators_all[operator_name])
    
        # calculate metrics of the current operator
        conf_matrix = confusion_matrix(cur_true_operator.numpy(), cur_pred_operator.numpy())
        accuracy = accuracy_score(cur_true_operator.numpy(), cur_pred_operator.numpy())
        precision = precision_score(cur_true_operator.numpy(), cur_pred_operator.numpy(), zero_division=1.0)
        recall = recall_score(cur_true_operator.numpy(), cur_pred_operator.numpy(), zero_division=0.0)
        f1 = f1_score(cur_true_operator.numpy(), cur_pred_operator.numpy())
    
        # store the metrics of the current operator
        metrics["operator"].append(operator_name)
        metrics["conf_matrix"].append(conf_matrix)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
    
    return metrics