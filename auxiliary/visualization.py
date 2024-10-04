import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.metrics import ConfusionMatrixDisplay

# Define DIN font for plots if working locally
if not torch.cuda.is_available():
    plt.rcParams["font.family"] = "DIN Alternate"

# define standard figure size for plots
figsize = (10, 6)

def assign_labelset(row, labels):
    '''Given a DataFrame row, returns a concatenated string of the specified column names, where the columns = 1.'''
    labelset = []

    for label in labels:
        if label in row and row[label] == 1:
            labelset.append(label.capitalize())

    if labelset:
        return ', '.join(labelset)

    return "None"

def plot_geometry(geom, ax, **kwargs):
    '''Visualizes a given shapely geometry on a given axis.'''
    if geom.geom_type == 'Point':
        ax.plot(geom.x, geom.y, 'o', **kwargs)
    elif geom.geom_type == 'LineString':
        x, y = geom.xy
        ax.plot(x, y, **kwargs)
    elif geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

def plot_raster(raster, axis=False, color=None):
    '''Visualizes a given raster'''
    # prepare figure and axis
    fig, ax = plt.subplots(1, figsize = (5,5))

    if not axis:
        ax.axis("off")

    # remove axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    # define the colormap
    if color:
        cmap = ListedColormap(['white', color])
    else:
        cmap = plt.cm.gray_r

    # plot the raster with specified colormap
    plt.imshow(raster, cmap=cmap, vmin=0, vmax=1)

def plot_graph(graph, ax, position=True, labels=False, node_color=None, node_color_map=None, edge_color=None, edge_color_map=None, axis=True):
    '''Visualizes a given graph on a given axis using the centroid coordinates associated with the nodes'''
    # extract the centroid coordinates for the nodes
    if position:
        x_coords = nx.get_node_attributes(graph, "coord_x")
        y_coords = nx.get_node_attributes(graph, "coord_y")

        # determine node positions according to centroid coordinates
        pos = {i: (x_coords[i], y_coords[i]) for i in range(graph.number_of_nodes())}
    else:
        pos = None

    if node_color:
        node_colors = [node_color_map[graph.nodes[n][node_color]] for n in graph.nodes()]
    else:
        node_colors = "k"

    if edge_color:
        edge_colors = [edge_color_map[graph.edges[e][edge_color]] for e in graph.edges()]
    else:
        edge_colors = "r"

    # draw the graph
    nx.draw(G=graph, pos=pos, ax=ax, node_size=100, width=4, edge_color=edge_colors, node_color=node_colors, arrows=False)

    # ensure the axes are visible
    if axis:
        ax.set_axis_on()

    # add labels if specified
    if labels:
        nx.draw_networkx_labels(G=graph, pos=pos, font_size=12, font_color="white")

def visualize_operator_distribution(buildings, operators, save=False, path=None, figsize=(10,6), display_ratio=False, display_legend=True, display_ylabel=True):
    '''Plots the distribution of the given generalization operators within a buildings DataFrame individually in a barplot.'''
    prevalences = {
        "not present": [],
        "present": []
    }
    
    color_map = {
        "present": "black",  # Black color for present
        "not present": "white"  # White color for not present
    }
        
    for operator in operators:
        n_present = (buildings[operator] == 1).sum()
        n_not_present = (buildings[operator] == 0).sum()
            
        prevalences["present"].append(n_present)
        prevalences["not present"].append(n_not_present)
    
    width = 0.5
    
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(operators))
    
    total_buildings = buildings.shape[0]
    
    for boolean, prevalence in prevalences.items():
        if display_ratio:
            prevalence = [count / total_buildings for count in prevalence]  # Convert counts to ratio
        p = ax.bar(operators, prevalence, width, label=boolean, bottom=bottom, edgecolor="black", color=color_map[boolean])
        bottom += prevalence
        
    for bar in ax.patches:
        if display_ratio:
            ax.text(x = bar.get_x() + bar.get_width() / 2,
                    y = bar.get_height() / 2 + bar.get_y(),
                    s = f'{bar.get_height() * 100:.1f}%', ha = 'center',
                    color = 'black' if bar.get_facecolor() != (0.0, 0.0, 0.0, 1.0) else 'white', weight = 'bold', size = 14)
        else:
            ax.text(x = bar.get_x() + bar.get_width() / 2,
                    y = bar.get_height() / 2 + bar.get_y(),
                    s = f'{bar.get_height() / buildings.shape[0] * 100:.1f}%', ha = 'center',
                    color = 'black' if bar.get_facecolor() != (0.0, 0.0, 0.0, 1.0) else 'white', weight = 'bold', size = 14)
    
    if display_ylabel:
        ax.set_ylabel('Number of buildings' if not display_ratio else 'Proportion of buildings', fontsize=15)
    
    # set the formatter for the y-axis to use non-scientific notation
    if not display_ratio:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    else:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # capitalize x-axis labels
    ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    if display_legend:
        ax.legend([handles[idx] for idx in order],
                  [labels[idx] for idx in order],
                  loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 14, 'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

    if save:
        fig.savefig(path, bbox_inches="tight")

def visualize_labelset_distribution(buildings, labels, save=False, path=None):
    '''Plots the distribution of the labelsets in a given buildings DataFrame in a barplot.'''
    # assign labelsets
    buildings = buildings.copy()
    buildings["labelset"] = buildings.apply(lambda row: assign_labelset(row, labels), axis=1)

    labelset_counts = buildings["labelset"].value_counts()
    labelset_counts_df = labelset_counts.reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(labelset_counts_df["labelset"], labelset_counts_df["count"], color="black")
    ax.set_xlabel("Number of buildings", fontsize=15)  # set the x-axis label
    
    # set the formatter for the y-axis to use non-scientific notation
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

    if save:
        fig.savefig(path, bbox_inches="tight")

def visualize_losses(loss_file, path_to_loss_files, save=False, output_path=None):
    '''Given the name of and path to a CSV file with training and validation loss, creates a graph that visualizes the loss curves.'''
    plt.figure(figsize=figsize)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)

    loss_file_path = os.path.join(path_to_loss_files, loss_file)
    cur_loss = pd.read_csv(loss_file_path)

    epochs = list(range(1, cur_loss.shape[0] + 1))
        
    cur_training_loss = cur_loss["training_loss"]
    cur_validation_loss = cur_loss["validation_loss"]
        
    plt.plot(epochs, cur_training_loss, color="orange", label="training loss")
    plt.plot(epochs, cur_validation_loss, color="red", label="validation loss")

    # Set the x-axis to use integer locator
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Only integer labels

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 14, 'weight': 'bold'})
    plt.show()

    if save:
        fig.savefig(output_path, bbox_inches="tight")

def visualize_multiple_losses(loss_files, path_to_data, model, epochs_every=5, save=False, output_path=None, figsize=(10,6)):
    '''Given the names of and paths to CSV files with training and validation losses, creates two figures that visualize 
    the training and validation loss curves respectively.'''
    assert model in ("raster", "vector", "multimodal")
    # path to the model outputs
    path_to_model_outputs = os.path.join(path_to_data, model, "model_outputs")

    # generate separate figures for training and validation loss
    fig_training, ax_training = plt.subplots(1, 1, figsize=figsize)
    fig_validation, ax_validation = plt.subplots(1, 1, figsize=figsize)

    # getting full operator names from abbreviations
    operator_map = {"eli": "elimination", "sel": "selection"}

    # colors to use for the losses
    colors = ["gold", "darkorange", "lightblue", "royalblue"]

    # initialize min and max loss values
    min_loss = np.inf
    max_loss = -np.inf

    # initialize max epoch
    max_epoch = -np.inf

    # early stopping point size
    point_size = 75

    for i, (loss_file, epochs_early_stopping) in enumerate(loss_files.items()):
        # determine architecture and operators from filename
        loss_file_split = loss_file.split("_")
        architecture = loss_file_split[0]
        operator = operator_map[loss_file_split[1]]

        # path to losses
        loss_file_path = os.path.join(path_to_model_outputs, operator, "losses", loss_file)
        cur_loss = pd.read_csv(loss_file_path)

        # get number of epochs
        epochs = list(range(1, cur_loss.shape[0]+1))

        # get training and validation losses
        cur_training_loss = cur_loss["training_loss"]
        cur_validation_loss = cur_loss["validation_loss"]

        # update min and max loss values
        min_loss = min(min_loss, cur_training_loss.min(), cur_validation_loss.min())
        max_loss = max(max_loss, cur_training_loss.max(), cur_validation_loss.max())

        # update max epoch
        max_epoch = max(max_epoch, max(epochs))

        # plot the losses on the respective figure
        ax_training.plot(epochs, cur_training_loss, label=f"{operator.capitalize()} with {architecture}", color=colors[i], zorder=1)
        ax_validation.plot(epochs, cur_validation_loss, label=f"{operator.capitalize()} with {architecture}", color=colors[i], zorder=1)

        # store the loss value at the epoch determined through early stopping
        loss_early_stopping_training = cur_loss[cur_loss["epoch"] == epochs_early_stopping]["training_loss"].item()
        loss_early_stopping_validation = cur_loss[cur_loss["epoch"] == epochs_early_stopping]["validation_loss"].item()

        ax_training.scatter(epochs_early_stopping, loss_early_stopping_training, color=colors[i], s=point_size, zorder=2)
        ax_validation.scatter(epochs_early_stopping, loss_early_stopping_validation, color=colors[i], s=point_size, zorder=2)

    # add dummy scatter plot for the legend
    ax_validation.scatter([], [], zorder=2, color="black", s=point_size, label="Early stopping")

    # set the same y-axis limits for both plots
    ax_training.set_ylim([min_loss-0.05 * min_loss, max_loss+0.05 * min_loss])
    ax_validation.set_ylim([min_loss-0.05 * min_loss, max_loss+0.05 * min_loss])

    # set x-axis limits for both plots
    ax_training.set_xlim([1, max_epoch+5])
    ax_validation.set_xlim([1, max_epoch+5])
    
    # set axis labels
    ax_training.set_xlabel("Epoch", fontsize=15)
    ax_training.set_ylabel("Loss", fontsize=15)
    ax_validation.set_xlabel("Epoch", fontsize=15)

    # set x-axis tick marks to appear every 5 epochs
    ax_training.xaxis.set_major_locator(MultipleLocator(epochs_every))
    ax_validation.xaxis.set_major_locator(MultipleLocator(epochs_every))

    # set axis parameters
    ax_training.tick_params(axis="both", which="major", labelsize=14)
    ax_training.spines["top"].set_visible(False)
    ax_training.spines["right"].set_visible(False)
    ax_validation.tick_params(axis="both", which="major", labelsize=14)
    ax_validation.spines["top"].set_visible(False)
    ax_validation.spines["right"].set_visible(False)

    ax_validation.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 14, 'weight': 'bold'})
    plt.show()

    if save:
        fig_training.savefig(os.path.join(output_path, f"losses_{model}_training.png"), bbox_inches="tight")
        fig_validation.savefig(os.path.join(output_path, f"losses_{model}_validation.png"), bbox_inches="tight")

def visualize_confusion_matrix(conf_matrix, operator):
    '''Visualizes a given confusion matrix.'''
    plt.rcParams.update({"font.size": 14})

    # display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Present", "Present"])
    disp.plot(cmap=plt.cm.Greys, colorbar=False)
    plt.xlabel(f"Predicted {operator}", fontsize=15)
    plt.ylabel(f"True {operator}", fontsize=15)
    plt.show()
