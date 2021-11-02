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
from hrga import *


def neighboring_graph_structure(G, gdf, n_origins, n_destinations):
    """
    Generate a graph based on neighboring topology for
    the geometry.
    
    Args:
        G - directed graph of the road network
        gdf - GeoDataFrame with geometry of administrative regions
        n_origins - list containing node_id of neighbor origins
        n_destinations - list containing node_id of neighbor destinations
        
    Returns: A simplified graph, containing only the nearest nodes to
    centroids and edges connecting neighboring geometries. The structure retains
    geometry for nodes, but not for edges.
    """
    
    print(f' ================================== ')
    print(f' === Neighboring Graph Function === ')
    print(f' ================================== ')
    print('')
    print('')
    print('Step 1: Calcuate Nearest Nodes for all Centroids')
    
    # Times
    st_time = time.time()
    nn_s_time = time.time()
    
    # For all centroids, find closest node
    nearest_nodes = ox.nearest_nodes(G, gdf.centroid_x.values, gdf.centroid_y.values)

    print(f'   Done. Step 1 execution time: {(time.time() - nn_s_time):.4f} seconds')
    print('')
    print('Step 2: Build Graph and Add Nodes')
    
    bg_s_time = time.time()    
    
    # Start Building New Graph
    sG = nx.Graph()

    # Add nodes
    # Keep osmnx id, but add a property 'node_id' to the node
    for i, node in enumerate(nearest_nodes):
        sG.add_node(i+1, osm_id=node, neighborhood_id=gdf.iloc[i].neighborhood_id, neighborhood=gdf.iloc[i].neighborhood, pos=(G.nodes[node]['x'], G.nodes[node]['y']))

    print(f'   Done. Step 2 execution time: {(time.time() - bg_s_time):.4f} seconds')
    print(f'')
    print(f'Step 3: Add Edges and Calculate Shortest Paths')
    
    ae_s_time = time.time()
    
    # Edge ID Iterator
    edge_id = 1
    
    # Add edges
    # Iterate over neighboring list
    for i, origin in enumerate(n_origins):
        
        it_s_time = time.time()
        
        # Calculate shortest path from n_origins[i] to n_destinations[i]
        # Calculate Shortest Path from 'i' to 'j'
        shortest_path_ij = nx.shortest_path(G, sG.nodes[n_origins[i]]['osm_id'], sG.nodes[n_destinations[i]]['osm_id'], weight='length')
        shortest_path_ij_length = nx.shortest_path_length(G, sG.nodes[n_origins[i]]['osm_id'], sG.nodes[n_destinations[i]]['osm_id'], weight='length')
        
        # Calculate Shortest Path from 'j' to 'i'
        shortest_path_ji = nx.shortest_path(G, sG.nodes[n_destinations[i]]['osm_id'], sG.nodes[n_origins[i]]['osm_id'], weight='length')
        shortest_path_ji_length = nx.shortest_path_length(G, sG.nodes[n_destinations[i]]['osm_id'], sG.nodes[n_origins[i]]['osm_id'], weight='length')
        
        # Calculate Avg Length (ij + ji / 2)
        shortest_path_avg_length = (shortest_path_ij_length + shortest_path_ji_length) / 2
        
        # Add edge
        sG.add_edge(
            n_origins[i],
            n_destinations[i],
            osm_origin=sG.nodes[n_origins[i]]['osm_id'],
            osm_destination=sG.nodes[n_destinations[i]]['osm_id'],
            length=shortest_path_avg_length,
            uv_sequence=shortest_path_ij,
            vu_sequence=shortest_path_ji,
            uv_length=shortest_path_ij_length,
            vu_length=shortest_path_ji_length,
        )                
        
        print(f'   Iteration {i+1}/{len(n_origins)} Done. Execution time: {(time.time() - it_s_time):.4f} seconds')
    
    print(f'   Done. Step 3 execution time: {(time.time() - ae_s_time):.4f} seconds')
    print(f'')
    print(f'Finished. Total execution time: {(time.time() - st_time):.4f} seconds')
    
    return sG



def neighboring_graph_structure_adm(G, gdf, n_origins, n_destinations):
    """
    Generate a graph based on neighboring topology for
    the geometry.
    
    Args:
        G - directed graph of the road network
        gdf - GeoDataFrame with geometry of administrative regions
        n_origins - list containing node_id of neighbor origins
        n_destinations - list containing node_id of neighbor destinations
        
    Returns: A simplified graph, containing only the nearest nodes to
    centroids and edges connecting neighboring geometries. The structure retains
    geometry for nodes, but not for edges.
    """
    
    print(f' ================================== ')
    print(f' === Neighboring Graph Function === ')
    print(f' ================================== ')
    print('')
    print('')
    print('Step 1: Calcuate Nearest Nodes for all Centroids')
    
    # Times
    st_time = time.time()
    nn_s_time = time.time()
    
    # For all centroids, find closest node
    nearest_nodes = ox.nearest_nodes(G, gdf.centroid_x.values, gdf.centroid_y.values)

    print(f'   Done. Step 1 execution time: {(time.time() - nn_s_time):.4f} seconds')
    print('')
    print('Step 2: Build Graph and Add Nodes')
    
    bg_s_time = time.time()    
    
    # Start Building New Graph
    sG = nx.Graph()

    # Add nodes
    # Keep osmnx id, but add a property 'node_id' to the node
    for i, node in enumerate(nearest_nodes):
        sG.add_node(i+1, osm_id=node, adm_region_id=gdf.iloc[i].adm_region_id, adm_region=gdf.iloc[i].adm_region, pos=(G.nodes[node]['x'], G.nodes[node]['y']))

    print(f'   Done. Step 2 execution time: {(time.time() - bg_s_time):.4f} seconds')
    print(f'')
    print(f'Step 3: Add Edges and Calculate Shortest Paths')
    
    ae_s_time = time.time()
    
    # Add edges
    # Iterate over neighboring list
    for i, origin in enumerate(n_origins):
        
        it_s_time = time.time()
        
        # Calculate shortest path from n_origins[i] to n_destinations[i]
        # Calculate Shortest Path from 'i' to 'j'
        shortest_path_ij = nx.shortest_path(G, sG.nodes[n_origins[i]]['osm_id'], sG.nodes[n_destinations[i]]['osm_id'], weight='length')
        shortest_path_ij_length = nx.shortest_path_length(G, sG.nodes[n_origins[i]]['osm_id'], sG.nodes[n_destinations[i]]['osm_id'], weight='length')
        
        # Calculate Shortest Path from 'j' to 'i'
        shortest_path_ji = nx.shortest_path(G, sG.nodes[n_destinations[i]]['osm_id'], sG.nodes[n_origins[i]]['osm_id'], weight='length')
        shortest_path_ji_length = nx.shortest_path_length(G, sG.nodes[n_destinations[i]]['osm_id'], sG.nodes[n_origins[i]]['osm_id'], weight='length')
        
        # Calculate Avg Length (ij + ji / 2)
        shortest_path_avg_length = (shortest_path_ij_length + shortest_path_ji_length) / 2
        
        # Add edge
        sG.add_edge(
            n_origins[i],
            n_destinations[i],
            osm_origin=sG.nodes[n_origins[i]]['osm_id'],
            osm_destination=sG.nodes[n_destinations[i]]['osm_id'],
            length=shortest_path_avg_length,
            uv_sequence=shortest_path_ij,
            vu_sequence=shortest_path_ji,
            uv_length=shortest_path_ij_length,
            vu_length=shortest_path_ji_length
        )                
                
        print(f'   Iteration {i+1}/{len(n_origins)} Done. Execution time: {(time.time() - it_s_time):.4f} seconds')
    
    print(f'   Done. Step 3 execution time: {(time.time() - ae_s_time):.4f} seconds')
    print(f'')
    print(f'Finished. Total execution time: {(time.time() - st_time):.4f} seconds')
    
    return sG



def generate_od_pair_list(G, demand_matrix):
    """
    Generates a list of origin-destination pairs based on
    the demand matrix.
    
    Args:
        G - graph of network
        demand_matrix - matrix with demand from 'o' to 'd'

    Returns: list of origin-destination pairs
    """
    
    # Initialize
    od_pairs = []
    
    # Iterate
    for origin in G.nodes():
        for destination in G.nodes():
            
            # If origin != destination and demand is greater than zero 
            if (origin != destination) and (demand_matrix[origin - 1][destination - 1] > 0):
                
                # Append to list
                od_pairs.append(f'{origin}-{destination}')
                
    return od_pairs



def k_shortest_paths(G, source, target, k, weight=None):
    """
    Generates k shortest paths from origin 'o' to destination 'd'.
    
    Args:
        G - graph of network
        source - origin
        target - destination
        k - max number of paths generated
        weight - property used to calculate path

    Returns: A list of the k shortest paths found
    """
    
    return list(itertools.islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def generate_paths(G, od_pairs, k=10, weight=None):
    """
    Generates k shortest paths from origin 'o' to destination 'd'
    for all origin destination pairs
    
    Args:
        G - graph of network
        od_pairs - list of origin-destination pairs
        k - max number of paths returned
        weight - parameter for shortest path calculation

    Returns: a tuple, containing a dictionaty with the number of paths 
    generated for each od-pair and a dictionary containing the paths itself.
    """
    
    # Initialize List
    paths = {}
    num_paths = {}

    # Iterate
    for od in od_pairs:
        
        # Set origin and destination
        origin = int(od.split('-')[0])
        destination = int(od.split('-')[1])
        
        # Calculate k shortest paths from 'o' to 'd'
        od_paths = k_shortest_paths(G, origin, destination, k, weight=weight)
        
        # Populate dictionary
        num_paths[od] = len(od_paths)
        paths[od] = od_paths

    return paths, num_paths


def generate_DELTA_matrix(G, routes):
    """
    Generates DELTA incidence matrix (1 if edge 'e' belongs
    to route 'r', and 0 otherwise).
    
    Args:
        G - graph of road network.
        routes - list of routes

    Returns: The incidence matrix DELTA
    """
    
    # Initialize matrix
    DELTA = np.zeros(( len(routes), len(G.edges()) ))
    
    # Iterate over all routes
    for r, route in enumerate(routes):
        
        # Get all edges in the route
        edges_in_route = path_nodes_to_edges(route)
        
        # Iterate over all edges
        for e, edge in enumerate(G.edges()):
            
            # If edge exists in route, change value to 1
            if (edge in edges_in_route) or (edge[::-1] in edges_in_route):
                DELTA[r][e] = 1
                
    return DELTA.astype(np.int32)


def generate_delta_matrix(G, od_pairs, num_paths, paths, k=10):
    """
    Generates delta incidence matrix (1 if edge 'e' belongs
    to path 'p' from origin 'o' to destination 'd').
    
    Args:
        G - graph of road network.
        od_pairs - list of origin-destination pairs
        num_pahts - dict containing number of paths from 'o' to 'd'
        paths - dict of shortest paths from 'o' to 'd'

    Returns: The incidence matrix delta
    """
    
    # Initialize matrix as empty list
    delta = []
    
    
    # Iterate over OD pairs
    for origin_destination in od_pairs:
        
        # Initialize paths vector
        od = []
        
        # Iterate over all k paths from O to D
        for p in range(k):
            
            # Initialize edge vector
            used_edges_in_path = []
            
            # If there is a kth path, do the procedure
            if p < len(paths[origin_destination]):
            
                # Get all edges in path
                edges_in_path = path_nodes_to_edges(paths[origin_destination][p])                

                # Iterate over all edges
                for e, edge in enumerate(G.edges()):

                    # If edge exists in path, append 1 to used edges
                    if (edge in edges_in_path) or (edge[::-1] in edges_in_path):
                        used_edges_in_path.append(1)

                    # If not, append 0
                    else:
                        used_edges_in_path.append(0)
            
            # If there isnt a kth path            
            else:
                
                # Iterate over all edges 
                for e, edge in enumerate(G.edges()):
                    
                    # Append zeros
                    used_edges_in_path.append(0)
                    
            # After iterating over all edges, append used edges to list of paths
            od.append(used_edges_in_path)
            
        # After iterating over all paths, append paths to delta
        delta.append(od)
    
    return delta


def unique_routes(routes):
    """
    Returns an list with unique routes for all nodes.
    This function achieves this by removing simetrical
    routes (A, B, C) == (C, B, A).

    Args:
        routes - a list of routes, as sequences of nodes
        
    Returns: An list with unique routes.
    """

    # Iterate over all routes in reversed order
    for route in list(reversed(routes)):
        
        # If the reversed route (C, B, A) exists in the list
        if list(reversed(route)) in routes:
            
            # Remove it from the list
            routes.remove(route)
            
    return routes