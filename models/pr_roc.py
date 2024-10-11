import sys
sys.path.append("models/raster")
sys.path.append("models/vector")
sys.path.append("models/multimodal")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from sklearn.metrics import auc

from metrics_raster import get_metrics_raster
from metrics_vector import get_metrics_vector
from metrics_multimodal import get_metrics_multimodal

def get_auc(x, y):
    '''Given two input arrays, sorts the values and computes the area under the curve.'''
    # ensure the inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # sort the x values and use the indices to sort y
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]
    
    # compute the AUC using the sorted values
    area_under_curve = auc(sorted_x, sorted_y)
    
    return area_under_curve

def get_closest_point(x, y, target_point):
    '''Given two input arrays with x and y coordinates, identifies the index of the closest point to target_point.'''
    # unpack the target point coordinates
    x_target, y_target = target_point

    # calculate the square of the Euclidean distance from each point to the target
    distances = (x - x_target)**2 + (y - y_target)**2

    # find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    return closest_index

def get_pr_roc(model, dataset, batch_size, operators_to_pred, device, interval, increment):
    '''For a trained model, get values for precision-recall and ROC curve by testing thresholds in the 
    specified interval with given increment.'''
    # get name of the model
    model_name = model.__class__.__name__
    if model_name not in ("CNN", "ViT", "HGT", "HGNN", "MultimodalModel"):
        raise NotImplementedError

    # generate thresholds to test
    num_increments = int((interval[1] - interval[0]) / increment) + 1
    thresholds = np.linspace(interval[0], interval[1], num=num_increments)

    # prepare dictionary with results
    metrics_dic = {"operator": [], "metric": []}
    for operator in operators_to_pred:
        metrics_dic["operator"].extend(3 * [operator])
        metrics_dic["metric"].append("precision")
        metrics_dic["metric"].append("recall")
        metrics_dic["metric"].append("fpr")

    for threshold in thresholds:
        metrics_dic[str(threshold)] = []

    # get metrics for every threshold
    for i, threshold in enumerate(thresholds):
        if model_name in ("CNN", "ViT"):
            metrics = get_metrics_raster(model=model, 
                                         dataset=dataset,
                                         batch_size=batch_size,
                                         operators_to_pred=operators_to_pred, 
                                         threshold=threshold, 
                                         device=device)
            
        elif model_name in ("HGT", "HGNN"):
            metrics = get_metrics_vector(model=model, 
                                         dataset=dataset,
                                         batch_size=batch_size,
                                         operators_to_pred=operators_to_pred, 
                                         threshold=threshold, 
                                         device=device)

        elif model_name == "MultimodalModel":
            metrics = get_metrics_multimodal(model=model, 
                                             dataset=dataset,
                                             batch_size=batch_size,
                                             operators_to_pred=operators_to_pred, 
                                             threshold=threshold, 
                                             device=device)

        # store precision, recall and false positive rate
        for i in range(len(operators_to_pred)):
            # precision
            cur_precision = metrics["precision"][i]
            metrics_dic[str(threshold)].append(cur_precision)

            # recall
            cur_recall = metrics["recall"][i]
            metrics_dic[str(threshold)].append(cur_recall)

            # false positive rate
            cur_conf_matrix = metrics["conf_matrix"][i]
            cur_fpr = cur_conf_matrix[0,1] / np.sum(cur_conf_matrix[0,:])
            metrics_dic[str(threshold)].append(cur_fpr)

        print(f"Threshold {i+1}/{len(thresholds)} processed.")

    # convert dictionary to DataFrame
    metrics_df = pd.DataFrame(metrics_dic)

    return metrics_df

def plot_pr_curve(pr_roc_files, validation, legend_order, figsize=(10,6), save=False, output_path=None):
    '''Given a list of files with accuracy metrics as output by get_pr_roc, plots the precision-recall curve.'''
    # concatenate all of the input files into one DataFrame
    metrics = pd.DataFrame()
    
    for pr_roc_file in pr_roc_files:
        cur_pr_roc_file = pd.read_csv(pr_roc_file)
        metrics = pd.concat([metrics, cur_pr_roc_file])

    # filter out necessary data for PR curves
    precision_recall_data = metrics[metrics["metric"].isin(["precision", "recall"])]

    # extract unique operators
    operators = precision_recall_data["operator"].unique()

    # get the thresholds
    thresholds = precision_recall_data.columns[2:]

    # initialize the figure
    fig, ax = plt.subplots(figsize=figsize)

    # define colors for the operators
    colors = ["tab:blue", "tab:brown", "tab:green", "blueviolet", "tab:orange"]

    # marker size for the optimal threshold
    markersize = 200

    # plot the PR curve for every operator
    for i, operator in enumerate(operators):
        operator_data = precision_recall_data[precision_recall_data["operator"] == operator]
        recall = operator_data[operator_data["metric"] == "recall"].iloc[:,2:].values.flatten()
        precision = operator_data[operator_data["metric"] == "precision"].iloc[:,2:].values.flatten()

        # calculate AUC
        operator_auc = get_auc(recall, precision)
        
        color = colors[i]
        if validation:
            # identify the optimal threshold: the threshold closest to (1,1)
            optimal_threshold_idx = get_closest_point(x=recall, y=precision, target_point=(1,1))

            ax.plot(recall, precision, marker="o", label=f"{operator.capitalize()} (threshold = {float(thresholds[optimal_threshold_idx]):.2f})", color=color)
            # plot the optimal threshold as a triangle
            ax.scatter(recall[optimal_threshold_idx], precision[optimal_threshold_idx], marker="^", color=color, s=markersize)
        else:
            ax.plot(recall, precision, marker="o", label=f"{operator.capitalize()} (AUC = {operator_auc:.2f})", color=color)

    # add the triangle to the legend
    if validation:
        ax.scatter([], [], marker="^", color="black", s=markersize, label="Optimal threshold")

    # reorder the legend with respect to the input
    handles, labels = ax.get_legend_handles_labels()
    order = [list(operators).index(operator) for operator in legend_order]
    
    if validation:
        order.append(len(order))

    # customize axes
    ax.set_xlabel("Recall", fontsize=15)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylabel("Precision", fontsize=15)
    ax.set_ylim(-0.05,1.05)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon=False, fontsize=12)

    plt.show()

    if save:
        fig.savefig(output_path, bbox_inches="tight")

def plot_roc_curve(pr_roc_files, validation, legend_order, figsize=(10,6), save=False, output_path=None):
    '''Given a list of files with accuracy metrics as output by get_pr_roc, plots the ROC curve.'''
    # concatenate all of the input files into one DataFrame
    metrics = pd.DataFrame()
    
    for pr_roc_file in pr_roc_files:
        cur_pr_roc_file = pd.read_csv(pr_roc_file)
        metrics = pd.concat([metrics, cur_pr_roc_file])

    # filter out necessary data for ROC curves
    fpr_tpr_data = metrics[metrics["metric"].isin(["fpr", "recall"])]

    # extract unique operators
    operators = fpr_tpr_data["operator"].unique()

    # get the thresholds
    thresholds = fpr_tpr_data.columns[2:]

    # initialize the figure
    fig, ax = plt.subplots(figsize=figsize)

    # define colors for the operators
    colors = ["tab:blue", "tab:brown", "tab:green", "blueviolet", "tab:orange"]

    # marker size for the optimal threshold
    markersize = 200

    # plot the ROC curve for every operator
    for i, operator in enumerate(operators):
        operator_data = fpr_tpr_data[fpr_tpr_data["operator"] == operator]
        fpr = operator_data[operator_data["metric"] == "fpr"].iloc[:,2:].values.flatten()
        tpr = operator_data[operator_data["metric"] == "recall"].iloc[:,2:].values.flatten()

        # calculate AUC
        operator_auc = get_auc(fpr, tpr)
        
        color = colors[i]
        if validation:
            # identify the optimal threshold: the threshold closest to (0,1)
            optimal_threshold_idx = get_closest_point(x=fpr, y=tpr, target_point=(0,1))

            ax.plot(fpr, tpr, marker="o", label=f"{operator.capitalize()} (threshold = {float(thresholds[optimal_threshold_idx]):.2f})", color=color)
            # plot the optimal threshold as a triangle
            ax.scatter(fpr[optimal_threshold_idx], tpr[optimal_threshold_idx], marker="^", color=color, s=markersize)
        else:
            ax.plot(fpr, tpr, marker="o", label=f"{operator.capitalize()} (AUC = {operator_auc:.2f})", color=color)

    # add the triangle to the legend
    if validation:
        ax.scatter([], [], marker="^", color="black", s=markersize, label="Optimal threshold")

    # reorder the legend with respect to the input
    handles, labels = ax.get_legend_handles_labels()
    order = [list(operators).index(operator) for operator in legend_order]
    
    if validation:
        order.append(len(order))

    # customize axes
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_ylim(-0.05,1.05)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon=False, fontsize=12)

    plt.show()

    if save:
        fig.savefig(output_path, bbox_inches="tight")