import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd
import itertools
import time
import random


def get_shortest_paths(G):
    """
    Generates a list containing values for shortest paths for all
    od-pairs in graph
    
    Args:
        G - graph of road network.
        
    Returns: shortest paths list
    """
    
    shortest_paths = {}
    
    for origin in list(G.nodes):
        for destination in list(G.nodes):
            
            if origin != destination:
                shortest_path = nx.shortest_path(G, origin, destination, weight='length')
                shortest_paths[f'{origin}-{destination}'] = shortest_path
    
    return shortest_paths


def path_nodes_to_edges(path):
    """
    Returns an ordered list of edges given a list of nodes
    
    Args:
        path - a path, as a sequence of nodes
        
    Returns: An ordered list of edges.
    """
    
    # Edge sequence initialization
    edge_sequence = []
    
    for i in range(len(path) - 1):
        edge_sequence.append((path[i], path[i+1]))
        
    return edge_sequence


def path_edges_to_nodes(path):
    """
    Returns an ordered list of nodes given a list of edges
    
    Args:
        path - a path, as a sequence of edges
        
    Returns: An ordered list of nodes.
    """
    
    # Edge sequence initialization
    node_sequence = [path[0][0]]
    
    for i in range(len(path)):
        node_sequence.append(path[i][1])
        
    return node_sequence


def reversed_edges(path):
    """
    Returns an list with edges reversed. e.g. [(1, 2), (2, 3)] returns
    [(2, 1), (3, 2)]. Used to check directionality of edges
    
    Args:
        path - a path, as a sequence of edges
        
    Returns: An list of reversed edges.
    """
    
    # Reversed initialization
    reversed_edges = []
    
    # Loop
    for edge in path:
        reversed_edges.append(edge[::-1])
        
    return reversed_edges


def check_all_nodes_covered(G, solution_set):
    """
    Checks if all nodes are covered by the solution set.
    
    Args:
        G - directed graph of road network.
        solution_set - current solution set (list of routes)
        
    Returns: Boolean, node not covered.
    """
    
    # List of nodes
    nodes_in_graph = list(G.nodes())
    
    # Nodes in solution_set
    edges_in_solution_set = [x for l in solution_set for x in l]
    nodes_in_solution_set = list(set(itertools.chain.from_iterable(edges_in_solution_set)))
    
    # Check for completeness
    for node in nodes_in_graph:
        if node not in nodes_in_solution_set:
            return False, [node]
    
    return True, []


def is_connected(solution_set):
    """
    Checks whether each rout in the solution set shares at least
    one node with one or more routes.
    
    Args:
        current_solution - current solution list of routes.

    Returns: boolean
    """
    
    # Iterate over all routes
    for route in solution_set:
        
        # Get nodes from route
        route_nodes = get_nodes_from_route(route)
        
        # Get nodes from all other routes
        other_nodes = get_nodes_from_all_routes_except(solution_set, route)
        
        # Iterate over nodes
        for node in route_nodes:
            
            # If node is not found, exit early
            if node not in other_nodes:
                return False
    
    # Else, return true
    return True