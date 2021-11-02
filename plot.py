import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd
import itertools
import time
import random

from utils import *

def draw_route(G, route, ax, width=5, color='k'):
    """
    Draws a route on a given ax
    
    Args:
        G - graph of network
        route - route, as sequence of nodes.
        ax - axis to draw

    Returns: draws route on ax.
    """
    
    # Get position for nodes
    G_positions = nx.get_node_attributes(G, 'pos')
    
    # Create graph for route
    routeG = nx.Graph()
    
    # Create Positions
    route_positions = [G_positions[node] for node in route]
    
    # Create Edges
    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    route_edges_length = [G[route[i]][route[i+1]]['length'] for i in range(len(route)-1)]
    
    # Put graph together
    for i, node in enumerate(route):
        routeG.add_node(node, pos=route_positions[i])
        
    for i in range(len(route_edges)):
        routeG.add_edge(*route_edges[i], length=route_edges_length[i])
    
    # Get route positions
    route_pos = nx.get_node_attributes(routeG, 'pos')
    
    # Plot Graph
    edges_plot = nx.draw_networkx_edges(routeG, route_pos, width=width, edge_color=color)