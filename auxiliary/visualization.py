import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def plot_raster(raster, axis=False):
    '''Visualizes a given raster'''
    # prepare figure and axis
    fig, ax = plt.subplots(1, figsize = (5,5))

    if not axis:
        ax.axis("off")

    # define the colormap
    cmap = plt.cm.gray_r

    # plot the raster with specified colormap
    plt.imshow(raster, cmap=cmap, vmin=0, vmax=1)

def plot_graph(graph, ax, position=True, labels=False, node_color=None, edge_color=None):
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
        node_color_map = {"building": "orange", "road": "red"}
        node_colors = [node_color_map[graph.nodes[n][node_color]] for n in graph.nodes()]
    else:
        node_colors = "k"

    if edge_color:
        edge_color_map = {"building to building": "orange", "building to road": "red"}
        edge_colors = [edge_color_map[graph.edges[e][edge_color]] for e in graph.edges()]
    else:
        edge_colors = "r"

    # draw the graph
    nx.draw(G=graph, pos=pos, ax=ax, node_size=30, width=2, edge_color=edge_colors, node_color=node_colors, arrows=False)

    # add labels if specified
    if labels:
        nx.draw_networkx_labels(G=graph, pos=pos, font_size=12, font_color="white")

def visualize_operator_distribution(buildings, operators, save=False, path=None):
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
    
    for boolean, prevalence in prevalences.items():
        p = ax.bar(operators, prevalence, width, label=boolean, bottom=bottom, edgecolor="black", color=color_map[boolean])
        bottom += prevalence
        
    for bar in ax.patches:
        ax.text(x = bar.get_x() + bar.get_width() / 2,
                y = bar.get_height() / 2 + bar.get_y(),
                s = f'{bar.get_height() / buildings.shape[0] * 100:.1f}%', ha = 'center',
                color = 'black' if bar.get_facecolor() != (0.0, 0.0, 0.0, 1.0) else 'white', weight = 'bold', size = 13)
    
    ax.set_ylabel('Number of buildings', fontsize=15)
    
    # set the formatter for the y-axis to use non-scientific notation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    
    # capitalize x-axis labels
    ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend([handles[idx] for idx in order],
              [labels[idx] for idx in order],
              loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 14, 'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

    if save:
        fig.savefig(path, bbox_inches="tight")

def visualize_labelset_distribution(buildings, save=False, path=None):
    '''Plots the distribution of the labelsets in a given buildings DataFrame in a barplot.'''
    labelset_counts = buildings["labelset"].value_counts()
    labelset_counts_df = labelset_counts.reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(labelset_counts_df["labelset"], labelset_counts_df["count"], color="black")
    ax.set_xlabel("Number of buildings", fontsize=15)  # set the x-axis label
    
    # set the formatter for the y-axis to use non-scientific notation
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

    if save:
        fig.savefig(path, bbox_inches="tight")

def visualize_losses(loss_files, path_to_loss_files, save=False, output_path=None):
    '''Given a list of files containing CSV files with training and validation loss, creates a graph that visualizes the loss curves.'''
    plt.figure(figsize=figsize)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)

    for loss_file in loss_files:
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

def visualize_confusion_matrix(conf_matrix, operator):
    '''Visualizes a given confusion matrix.'''
    plt.rcParams.update({"font.size": 14})

    # display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Present", "Present"])
    disp.plot(cmap=plt.cm.Greys, colorbar=False)
    plt.xlabel(f"Predicted {operator}", fontsize=15)
    plt.ylabel(f"True {operator}", fontsize=15)
    plt.show()
