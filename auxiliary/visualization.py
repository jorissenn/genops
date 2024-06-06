import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import networkx as nx

# define DIN font
plt.rcParams["font.family"] = "DIN Alternate"

# define standard figure size for plots
figsize = (10, 6)

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

def plot_graph(graph, ax, labels=False, node_color=None):
    '''Visualizes a given graph on a given axis using the centroid coordinates associated with the nodes'''
    # extract the centroid coordinates for the nodes
    x_coords = nx.get_node_attributes(graph, "coord_x")
    y_coords = nx.get_node_attributes(graph, "coord_y")

    # determine node positions according to centroid coordinates
    pos = {i: (x_coords[i], y_coords[i]) for i in range(graph.number_of_nodes())}

    if node_color:
        # get node types and create a color map
        types = nx.get_node_attributes(graph, node_color)
        unique_types = set(types.values())
        color_map = {t: i for i, t in enumerate(unique_types)}
        node_colors = [color_map[types[node]] for node in graph.nodes()]
    else:
        node_colors = "k"

    # draw the graph
    nx.draw(G=graph, pos=pos, ax=ax, with_labels=labels, node_size=20, width=2, edge_color="r", node_color=node_colors, arrows=False)

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
              loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=False, 
              prop={'size': 14, 'weight': 'bold'})
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
