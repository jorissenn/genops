import matplotlib.pyplot as plt
import networkx as nx

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
