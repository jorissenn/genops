import sys
sys.path.append("models/raster")
sys.path.append("models/vector")
sys.path.append("models/multimodal")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from sklearn.metrics import roc_curve, auc

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
    num_increments = math.ceil((interval[1] - interval[0]) / increment) + 1
    thresholds = np.linspace(interval[0], interval[1], num=num_increments)
    thresholds = [round(threshold, 3) for threshold in thresholds]

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
    for threshold_idx, threshold in enumerate(thresholds):
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

        print(f"Threshold {threshold_idx+1}/{len(thresholds)} processed.")

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
        operator_data = precision_recall_data[precision_recall_data["operator"] == operator].dropna(axis=1, how="any")
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
        operator_data = fpr_tpr_data[fpr_tpr_data["operator"] == operator].dropna(axis=1, how="any")
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

def create_model_operator_roc_subplots(figsize, padding, orientation="vertical"):
    '''Given a figure size and a padding, creates a figure with 5 x 3 subplots that can be used to plot ROC curves 
    by model and operator.'''
    assert orientation in ("vertical", "horizontal")
    
    if orientation == "vertical":
        # create 5x3 subplots
        fig, ((ax_eli_raster, ax_eli_vector, ax_eli_multimodal),
              (ax_agg_raster, ax_agg_vector, ax_agg_multimodal),
              (ax_typ_raster, ax_typ_vector, ax_typ_multimodal),
              (ax_dis_raster, ax_dis_vector, ax_dis_multimodal),
              (ax_enl_raster, ax_enl_vector, ax_enl_multimodal)) = plt.subplots(nrows=5, ncols=3, figsize=figsize)
        fig.tight_layout(pad=padding)
        
        # set models as figure title
        model_fontsize = 20
        ax_eli_raster.set_title("Raster", fontsize=model_fontsize)
        ax_eli_vector.set_title("Vector", fontsize=model_fontsize)
        ax_eli_multimodal.set_title("Multimodal", fontsize=model_fontsize)
        
        # set axis labels at the margins
        axis_fontsize = 13
        ax_enl_raster.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_enl_vector.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_enl_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_eli_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_agg_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_typ_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_dis_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_enl_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
    
        # set operators as secondary y-axes
        operator_fontsize = 20
        elimination_position = ax_eli_multimodal.get_position()
        fig.text(elimination_position.x1, elimination_position.y0+elimination_position.height/2, 
                 "Elimination", va="center", ha="left", fontsize=operator_fontsize, rotation=90)
        aggregation_position = ax_agg_multimodal.get_position()
        fig.text(aggregation_position.x1, aggregation_position.y0+aggregation_position.height/2, 
                 "Aggregation", va="center", ha="left", fontsize=operator_fontsize, rotation=90)
        typification_position = ax_typ_multimodal.get_position()
        fig.text(typification_position.x1, typification_position.y0+typification_position.height/2, 
                 "Typification", va="center", ha="left", fontsize=operator_fontsize, rotation=90)
        displacement_position = ax_dis_multimodal.get_position()
        fig.text(displacement_position.x1, displacement_position.y0+displacement_position.height/2, 
                 "Displacement", va="center", ha="left", fontsize=operator_fontsize, rotation=90)
        enlargement_position = ax_enl_multimodal.get_position()
        fig.text(enlargement_position.x1, enlargement_position.y0+enlargement_position.height/2, 
                 "Enlargement", va="center", ha="left", fontsize=operator_fontsize, rotation=90)

    elif orientation == "horizontal":
        # create 3x5 subplots
        fig, ((ax_eli_raster, ax_agg_raster, ax_typ_raster, ax_dis_raster, ax_enl_raster), 
              (ax_eli_vector, ax_agg_vector, ax_typ_vector, ax_dis_vector, ax_enl_vector),
              (ax_eli_multimodal, ax_agg_multimodal, ax_typ_multimodal, ax_dis_multimodal, ax_enl_multimodal)) = plt.subplots(nrows=3, 
                                                                                                                              ncols=5, 
                                                                                                                              figsize=figsize)
        fig.tight_layout(pad=padding)
        
        # set operators as figure title
        operator_fontsize = 20
        ax_eli_raster.set_title("Elimination", fontsize=operator_fontsize)
        ax_agg_raster.set_title("Aggregation", fontsize=operator_fontsize)
        ax_typ_raster.set_title("Typification", fontsize=operator_fontsize)
        ax_dis_raster.set_title("Displacement", fontsize=operator_fontsize)
        ax_enl_raster.set_title("Enlargement", fontsize=operator_fontsize)
        
        # set axis labels at the margins
        axis_fontsize = 13
        ax_eli_raster.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_eli_vector.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_eli_multimodal.set_ylabel("True positive rate", fontsize=axis_fontsize)
        ax_eli_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_agg_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_typ_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_dis_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
        ax_enl_multimodal.set_xlabel("False positive rate", fontsize=axis_fontsize)
    
        # set models as secondary y-axes
        model_fontsize = 20
        raster_position = ax_enl_raster.get_position()
        fig.text(raster_position.x1, raster_position.y0+raster_position.height/2, 
                 "Raster", va="center", ha="left", fontsize=model_fontsize, rotation=90)
        vector_position = ax_enl_vector.get_position()
        fig.text(vector_position.x1, vector_position.y0+vector_position.height/2, 
                 "Vector", va="center", ha="left", fontsize=model_fontsize, rotation=90)
        multimodal_position = ax_enl_multimodal.get_position()
        fig.text(multimodal_position.x1, multimodal_position.y0+multimodal_position.height/2, 
                 "Multimodal", va="center", ha="left", fontsize=model_fontsize, rotation=90)
    
    return fig, ((ax_eli_raster, ax_eli_vector, ax_eli_multimodal),
                 (ax_agg_raster, ax_agg_vector, ax_agg_multimodal),
                 (ax_typ_raster, ax_typ_vector, ax_typ_multimodal),
                 (ax_dis_raster, ax_dis_vector, ax_dis_multimodal),
                 (ax_enl_raster, ax_enl_vector, ax_enl_multimodal))

def plot_roc_by_category_on_axis(df, true_label_col, pred_score_col, category_col, ax, colors, label_size=8):
    '''Given a DataFrame df and the column names of the true labels, the predicted scores and the categorical variable,
    plots an ROC curve for the observations belonging to each category on the provided axis.'''
    # extract categories
    categories = np.unique(df[category_col])

    # collect AUCs to sort individual curves in the legend
    aucs = []

    for i, category in enumerate(categories):
        # extract all observations belonging to the category
        df_category = df[df[category_col] == category]

        # extract the true labels and predicted scores
        true_labels = df_category[true_label_col].to_numpy()
        pred_scores = df_category[pred_score_col].to_numpy()

        # calculate ROC values
        fpr, tpr, thresholds = roc_curve(y_true=true_labels, 
                                         y_score=pred_scores)

        # calculate and save AUC
        area_under_curve = auc(fpr, tpr)
        aucs.append(area_under_curve)

        # plot ROC curve
        ax.plot(fpr, tpr, label=f"{category} (AUC = {area_under_curve:.2f})", color=colors[i], linewidth=0.75)

    # reorder the legend with respect to descending AUC
    handles, labels = ax.get_legend_handles_labels()
    order = [aucs.index(auc) for auc in sorted(aucs, reverse=True)]

    # customize axes
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon=False, fontsize=label_size)
