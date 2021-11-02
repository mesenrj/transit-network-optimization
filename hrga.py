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

def route_generation_algorithm(G, D, num_routes, minimum_nodes, maximum_nodes):
    """
    Generates a set of routes for a transit network design optimization problem.
    
    Args:
        G - directed graph of road network.
        D - demand matrix for all nodes.
        num_routes - number of routes to be generated.
        minimum_nodes - minimum number of nodes in any route.
        maximum_nodes - maximum number of nodes in any route.

    Returns: The best set of routes.
    """
    
    # Initialize
    current_best_solution_set = initialize_routes(G, D, num_routes, minimum_nodes, maximum_nodes)
    
    # Calculate Score
    current_score = evaluate_routes(current_best_solution_set)
    
    # Local search
    solution_set, score = local_search(current_best_solution_set)
    
    return solution_set, score



def initialize_routes(G, D, num_routes, minimum_nodes, maximum_nodes, method='edges'):
    """
    Generates a initial set of routes for the route generation algorithm.
    
    Args:
        G - directed graph of road network.
        D - demand matrix for all nodes.
        num_routes - number of routes to be generated.
        minimum_nodes - minimum number of nodes in any route.
        maximum_nodes - maximum number of nodes in any route.

    Returns: A set of routes.
    """
    
    # Initialize solution list
    solution_list = []
    
    # Compute shortest paths between all origin-destination pairs
    shortest_paths = get_shortest_paths(G)
    
    # Compute edge usage probability
    G = get_edge_usage_statistics(get_edge_usage(G, D, shortest_paths))
    
    # While number of routes hasn't been exceeded
    print(f'Initializing Route Generation Algorithm...')
    print('')
    print(f'=======> Generating {num_routes} Routes...')
    
    # Counter
    i = 0
    
    while i < num_routes:
        
        # Initialize new route
        new_route = []
        
        # start generating new route by picking random edge
        new_route.append(get_random_edge(G))
        
        # pre determine number of nodes
        node_number = random.randint(minimum_nodes, maximum_nodes)
        
        # Add to count
        i += 1
        
        # Reset Counter
        j = 0
        
        # While number of nodes in line hasn't been exceeded
        while j < node_number -1:
            
            # Append or prepend edge based on graph topology and current route
            new_route = extend_route(G, new_route)
            
            # Find all shortest paths that contains at least one edge from the newly generated route
            #updated_shortest_paths = update_shortest_path_list(shortest_paths, new_route)
            
            # Calculate usage probabilities
            #G = get_edge_usage_statistics(get_edge_usage(G, D, updated_shortest_paths))
            
            # Add to count
            j += 1
        
        # Add route to solution list
        solution_list.append(new_route)
    
    # Check for completeness
    complete, missing_node = check_all_nodes_covered(G, solution_list)
    
    # If solution not complete, add route to make it complete and try again
    if not complete:
        print(f'Solution is not complete: missing node {missing_node}')
    
    # Check for connectedness
    connected = True
    
    # If routes unconnected
    if not connected:
        print(f'Solution is not fully connected: route {"a"} does not share any node with any other route')
    
    # Convert to Node Routes
    node_routes = []
    unduplicated = []
    
    for route in solution_list:
        node_routes.append(path_edges_to_nodes(route))
        
    # Check for duplicates
    # Iterate over all routes in reversed order
    for route in node_routes:
        
        # If the reversed route (C, B, A) exists in the list
        # Or if route (A, B, C) exists in the list
        if (list(reversed(route)) not in unduplicated) and (route not in unduplicated):
            
            # Remove it from the list
            unduplicated.append(route)

    # Convert back to Edge Routes
    initialization = []
    
    for route in unduplicated:
        initialization.append(path_nodes_to_edges(route))
    
    initialization.sort()
    
    
    if method == 'nodes':
        unduplicated.sort()
        
        print(f'=======> {len(unduplicated)}  of {num_routes} routes generated successfully!')
        print(f'=======> A total of {num_routes - len(unduplicated)} routes were duplicated and removed')
        print('')
        for i, route in enumerate(unduplicated):
            print(f'Route {i+1}: {route}')

        return unduplicated
        
        
    else:

        print(f'=======> {len(initialization)}  of {num_routes} routes generated successfully!')
        print(f'=======> A total of {num_routes - len(initialization)} routes were duplicated and removed')
        print('')
        for i, route in enumerate(initialization):
            print(f'Route {i+1}: {route}')

        return initialization



def evaluate_routes(G, solution_set, demand_matrix, alpha=1, beta=5):
    """
    Calculates score of solution given 
    
    Args:
        G - graph of the network.
        solution_set - current solution list of routes.
        demand_matrix - demand matrix.
        alpha - weight of distance minimization
        beta - weight of transfer minimization

    Returns: score
    """
    
    # Initialize variables
    route_cost = 0
    transfer_cost = 0
    
    # Convert routes to node sequence
    converted_routes = []

    for route in solution_set:
        converted_routes.append(path_edges_to_nodes(route))
    
    # Generate extended graph
    eG = create_extended_graph(G, converted_routes, solution_set, transfer_weight=0)
    
    for origin in list(G.nodes()):
        for destination in list(G.nodes()):            
            if demand_matrix[origin-1][destination-1] != 0:
                
                st_time = time.time()
            
                # Calculate Shortest Path
                shortest_path = nx.shortest_path(eG, origin, destination, weight='length')
                shortest_path_length = nx.shortest_path_length(eG, origin, destination, weight='length')

                # Add to costs
                route_cost += demand_matrix[origin-1][destination-1] * shortest_path_length
                transfer_cost += demand_matrix[origin-1][destination-1] * calculate_transfers(shortest_path)
                
                #print(f'Node {origin} to {destination} evaluation complete. Took {(time.time() - st_time):.4f} seconds.')

    # Objective function
    score = (alpha * route_cost) + (beta * transfer_cost)
    
    return score, (route_cost, transfer_cost)



def hill_climbing(G, D, current_solution, num_routes, minimum_nodes, maximum_nodes, iterations=100000):
    """
    Local search algorithm to improve routes.
    
    Args:
        G - directed graph of road network.
        D - demand matrix for all nodes.
        num_routes - number of routes to be generated.
        minimum_nodes - minimum number of nodes in any route.
        maximum_nodes - maximum number of nodes in any route.

    Returns: An improved set of routes.
    """
    
    # Initialize score
    solution_set_score = np.inf
    
    # Run 100k iterations
    for i in range(iterations):
        # Get current score of solutions
        current_score, _ = evaluate_routes(G, current_solution, D)
        
        # If current solution has a lower score than the best solution
        if solution_set_score > current_score:
            
            # Current solution is best solution
            solution_set = current_solution
            
            # Current solution score is best solution score
            solution_set_score = current_score
            
        # Otherwise
        else:
            
            # Best solution replaces current solution
            current_solution = solution_set
                
        # Apply modification
        current_solution = modify_solution(G, solution_set)
        
        print(f'Iteration {i+1}/{iterations}: Score: {solution_set_score} | Route Count: {len(current_solution)}')
    
    return solution_set


def tabu_search(G, D, num_routes, minimum_nodes, maximum_nodes, current_solution, max_tabu_size):
    """
    Tabu search algorithm to improve routes.
    
    Args:
        G - directed graph of road network.
        D - demand matrix for all nodes.
        num_routes - number of routes to be generated.
        minimum_nodes - minimum number of nodes in any route.
        maximum_nodes - maximum number of nodes in any route.
        max_tabu_size - maximum size of tabu list

    Returns: An improved set of routes.
    """
    
    # Calculate solution score
    solution_score = evaluate_routes(current_solution)
    
    # Generate Tabu List
    tabu_list = []
    
    # Iterate 5000 times
    for i in range(5000):
        
        # Generate candidate list
        candidate_list = []
        
        # Get 20 new candidates
        while len(candidate_list) < 20:
            
            # Modify solution
            new_solution_set = modify_solution(current_solution)
            
            # If new solution is not on tabu list and is connected
            if (new_solution_set not in tabu_list) and (is_connected(new_solution_set)):
                
                # Add to tabu list
                candidate_list.append(new_solution_set)
                
        
        # get best candidate from candidate list
        best_candidate = get_best_candidate(candidate_list)
        
        # get score
        new_score = evaluate_routes(best_candidate)
        
        # decision criteria
        if new_score < current_score:
            
            # current solution is replaced by new solution
            current_solution = best_candidate
            
            # score is replaced
            solution_score = new_score
            
            # tabu list is modified
            tabu_list = difference(current_solution, tabu_list)
            
            # tabu list expire
            tabu_list = expire(tabu_list, max_tabu_size)
            
    return current_solution            


def get_edge_usage(graph, demand_matrix, shortest_paths):
    """
    Calculates usage for each edge (in absolute values) of the graph.
    The algorithm considers that all users choose the shortest paths
    between their origin and destination.
    
    Args:
        graph - directed graph of road network.
        D - demand matrix for all nodes.
        shortest_paths - shortest_path ditctionary for all O-D pairs

    Returns: Edge usage for all edges.
    """
    
    # Initialize edge usage
    for edge in list(graph.edges()):
        nx.set_edge_attributes(graph, {edge : { "usage" : 0 }})
    
    for origin in list(graph.nodes()):
        for destination in list(graph.nodes()):
            if origin != destination:
                
                # Get shortest path sequence
                shortest_path = shortest_paths[f'{origin}-{destination}']
                
                # Iterate
                for i in range(len(shortest_path) - 1):
                    
                    # Simplify names
                    u = shortest_path[i]
                    v = shortest_path[i+1]
                    
                    # Variables
                    current_edge = (u, v)
                    current_usage = graph[u][v]['usage']
                    
                    demand = demand_matrix[origin-1][destination-1]
                    
                    # Set usage as current usage + demand from origin to destination
                    nx.set_edge_attributes(graph, {current_edge : { "usage" : current_usage + demand }})
    
    return graph


def get_edge_usage_statistics(graph):
    """
    Calculates edge usage statistics for each edge of the graph.
    The usage statistic is calculated by taking absolute edge usage,
    dividing by its weight (distance or travel time) and normalizing
    values between 0 and 1.
    
    Args:
        G - directed graph of road network with 'length' and 'usage'
        attributes

    Returns: Edge usage statistics for all edges.
    """
    
    # Initialize edge usage statistics
    for edge in list(graph.edges):
        nx.set_edge_attributes(graph, {edge : { "usage_stat" : 0 }})
    
    # Get edges
    edges = [(u, v) for u, v in graph.edges()]
    
    # Get usage and weights
    edge_usage = np.array([graph[u][v]['usage'] for u, v in graph.edges()], dtype=object)
    weights = np.array([graph[u][v]['length'] for u, v in graph.edges()], dtype=object)
    
    # Divide edge_usage by weight
    edge_usage_stat = edge_usage / weights
    
    # Normalize values
    norm_edge_usage_stat = edge_usage_stat / edge_usage_stat.sum()

    # Add values to graph
    for i, edge in enumerate(edges):
        nx.set_edge_attributes(graph, {edge : { "usage_stat" : norm_edge_usage_stat[i] }})
    
    return graph


def get_random_edge(G):
    """
    Chooses random edge based on edge usage statistics.
    
    Args:
        G - directed graph of road network with usage_stat attribute
        
    Returns: A random edge.
    """
    
    # List of edges and usage statistics   
    edges = [(u, v) for u, v in G.edges()]
    usage_stat = [G[u][v]['usage_stat'] for u, v in G.edges()]
    
    # Select random edge based on statistics
    return random.choices(edges, weights=usage_stat)[0]


def extend_route(G, route):
    """
    Prepends or appends edge to current route.
    
    Args:
        G - directed graph of road network with usage_stat attribute
        route - list containing tuples of edges in the graph.
        
    Returns: An extended route.
    """
    
    # Get vars
    first_node = route[0][0]
    last_node = route[-1][1]
    
    # Candidate edges
    prepend_candidates = list(G.edges(first_node))
    append_candidates = list(G.edges(last_node))
    
    # Remove first and last edge from candidates
    try:
        prepend_candidates.remove(route[0])    
    except:
        prepend_candidates.remove(route[0][::-1])
    
    try:
        append_candidates.remove(route[-1])  
    except:
        append_candidates.remove(route[-1][::-1])
    
    
    # Convert Route to Nodes
    route_nodes = path_edges_to_nodes(route)    
    
    # Remove edges already used   
    for edge in prepend_candidates:
        
        # If destination node already exists in route,
        # remove from candidates to avoid cycle
        if (edge in route) or (edge[1] in route_nodes):            
            prepend_candidates.remove(edge)

        elif edge[::-1] in route:
            prepend_candidates.remove(edge)
                
    for edge in append_candidates:
        
        # If destination node already exists in route,
        # remove from candidates to avoid cycle
        if (edge in route) or (edge[1] in route_nodes):            
            append_candidates.remove(edge)
           
        elif edge[::-1] in route:
            append_candidates.remove(edge)


    
    # Get length of arrays
    prep_count = len(prepend_candidates)
    app_count = len(append_candidates)
    
    # If no candidates, exit early
    if (prep_count == 0) and (app_count == 0):
        return route
    
    # Get usage probabilities
    prepend_probabilities = [G[u][v]['usage_stat'] for u, v in prepend_candidates]
    append_probabilities = [G[u][v]['usage_stat'] for u, v in append_candidates]
    
    # Decision list
    candidates = prepend_candidates + append_candidates
    probabilites = prepend_probabilities + append_probabilities
    
    # Select random edge based on statistics
    selected_edge = random.choices(candidates, weights=probabilites)
    selected_edge = selected_edge[0]
    

    # If selected edge in append list, append. Else, prepend
    if selected_edge in append_candidates:
        route.append(selected_edge)
    else:
        route.insert(0, selected_edge[::-1])
        
    return route


def update_shortest_path_list(shortest_paths, new_route):
    """
    Updates shortest paths list based on new route. Only
    shortest paths containing one or more edges from the route
    remain on the list.
    
    Args:
        shortest_paths - dictionary of shortest paths for all od pairs
        new_route - route in current construction
        
    Returns: An updated shortest path dictionary.
    """
    
    # Invert every element from route to check both directions
    rev_edges = reversed_edges(new_route)
    
    # Create list to check disjunction
    edges_in_route = new_route + rev_edges
    
    # For every od-pair in shortest path dict
    for od in shortest_paths:
        
        # Get sequence of edges
        edges_in_path = path_nodes_to_edges(shortest_paths[od])
        
        if set(edges_in_path).isdisjoint(edges_in_route):
            shortest_paths[od] = []
            
    return shortest_paths


def insert_node(G, route):
    """
    Inserts a random node in a route to modify it.
    
    Args:
        G - graph of the network
        route - the route, as a sequence of edges.

    Returns: A modified route, with an extra node.
    """

    # Initialize list
    removed_edge_index = []
    ins_candidate_edges = []
    ins_candidate_nodes = []
    
    # Search viable nodes
    for i, edge in enumerate(route):
        
        # Generate list of nodes 'k' as (u, k) and (k, v)
        in_nodes = [v for u, v in G.edges(edge[0])]
        out_nodes = [v for u, v in G.edges(edge[1])]

        # Generate intersection
        candidate_nodes = list(set(in_nodes) & set(out_nodes))
        
        # If len(candidate_nodes) > 0...
        if len(candidate_nodes) > 0:
            removed_edge_index.append(i)
            ins_candidate_edges.append(edge)
            ins_candidate_nodes.append(candidate_nodes)

    # Check for a viable solution
    if len(ins_candidate_edges) and len(ins_candidate_nodes) > 0:
        
        # Select random edge from candidates
        selection_index = random.randint(0, len(removed_edge_index) - 1)
        selected_edge = ins_candidate_edges[selection_index]
        selected_edge_index_in_route = removed_edge_index[selection_index]
        
        # Select random node from candidates for selected edge
        selected_node = random.choice(ins_candidate_nodes[selection_index])

        # Generate new edges to add
        edge_1 = (selected_edge[0], selected_node)
        edge_2 = (selected_node, selected_edge[1])
               
        # Remove edge from route
        route.pop(selected_edge_index_in_route)
        
        # Add new edges
        route.insert(selected_edge_index_in_route, edge_1)
        route.insert(selected_edge_index_in_route + 1, edge_2)
        
    return route


def remove_node(G, route):
    """
    Removes a random node from a route to modify it.
    
    Args:
        G - graph of the network
        route - the route, as a sequence of edges.

    Returns: A modified route, with one less node.
    """
    
    # Initialize lists
    candidate_edges_indexes = []
    candidate_new_edge = []
    
    # Iterate through all nodes
    for i in range(len(route) - 1):
        
        # Determine origin and destination
        origin = route[i][0]
        destination = route[i+1][1]
        
        # If node is removable
        if G.has_edge(origin, destination):
            
            # Add edge sequence indexes
            candidate_edges_indexes.append((i, i+1))
            candidate_new_edge.append((origin, destination))
            
    
    # Check for a viable solution
    if len(candidate_edges_indexes) and len(candidate_new_edge) > 0:
                
        # Select random node from candidates
        selected_node = random.choice(candidate_edges_indexes)
        selected_node_index = candidate_edges_indexes.index(selected_node)
        
        # Remove node from sequence
        route.pop(selected_node[1])
        route.pop(selected_node[0])
        
        # Add new edge
        route.insert(selected_node[0], candidate_new_edge[selected_node_index])
        
    return route


def swap_node(G, route):
    """
    Swaps a random node from a route to modify it.
    
    Args:
        G - graph of the network
        route - the route, as a sequence of edges.

    Returns: A modified route, with the same number of
    nodes but a different combination.
    """
    
    # Initialize lists
    swap_candidates_indexes = []
    swap_candidates = []
    
    # Transform path into node notation
    node_route = path_edges_to_nodes(route)
    
    # Iterate through all nodes
    for i in range(len(route) - 1):
        
        # Determine origin and destination
        origin = route[i][0]
        destination = route[i+1][1]
        
        # Collection of unused nodes
        unused_nodes = list(set(G.nodes()) - set(node_route))
               
        # if exists an alternative path
        for candidate_node in unused_nodes:
        
            # if exists a node that has edges from origin and to destination
            if G.has_edge(origin, candidate_node) and G.has_edge(candidate_node, destination):
            
                # Add edge sequence indexes
                swap_candidates_indexes.append((i, i+1))
                swap_candidates.append(((origin, candidate_node),(candidate_node, destination)))
                
    
    # Check for viable solution
    if len(swap_candidates_indexes) > 0 and len(swap_candidates) > 0:
        
        # Get length of list
        selected = random.randint(0, len(swap_candidates_indexes) - 1)
        
        # Select random edges to replace
        removed_edges = swap_candidates_indexes[selected]
        inserted_edges = swap_candidates[selected]
        
        # Remove old edges
        route.pop(removed_edges[1])
        route.pop(removed_edges[0])
        
        # Add new edges
        route.insert(removed_edges[0], inserted_edges[0])
        route.insert(removed_edges[1], inserted_edges[1])
        
    return route


def modify_solution(G, current_solution):
    """
    Tabu search algorithm to improve routes.
    
    Args:
        G - graph of the network
        current_solution - current solution list of routes.

    Returns: An modified set of routes.
    """
   
    # List of operations
    operations = [insert_node, remove_node, swap_node, 'remove_route']  
    
    # Select random route from list
    route_index = random.randint(0, len(current_solution) - 1)
    route = current_solution[route_index]
    
    # Select random operation
    operation = random.choice(operations)
    
    # Execute operation
    if operation == 'remove_route':
        # Delete route from array
        current_solution.pop(route_index)
        
    else:
        # Modify route
        modified_route = operation(G, route)
    
        # Remove old route from list
        current_solution.pop(route_index)

        # Add modified route to list
        current_solution.insert(route_index, modified_route)
    
    #print(f'Selected operation: {operation.__name__}')
    
    return current_solution


def create_extended_graph(graph, route_set, original_set, transfer_weight=0):

    extended_graph = nx.Graph()
    indexes = dict([(node, 0) for node in graph.nodes()])
    
    for i, route in enumerate(route_set):
        #print('')
        #print(f'Route {i+1}: {route}')
        #print(f'Route {i+1}: {original_set[i]}')
        for j, node in enumerate(route):
            current_node = 'R' + str(i+1) + '-' + str(node)
            
            # Set index for nodes
            current_node_index = node
            
            # Add node to graph
            extended_graph.add_node(current_node, original_node=node, route=i+1)
            
            # If its not the starting node, add the edge connecting to previous node
            if j > 0:
                tup = tuple(sorted((previous_node_index, current_node_index)))
                
                #print(f'previous_node: {previous_node}')
                #print(f'current_node: {current_node}')
                #print(f'graph: {graph}')
                #print(f'tup: {tup}')
                
                extended_graph.add_edge(current_node, previous_node, length=nx.get_edge_attributes(graph, 'length')[tup])
            
            # Update variables
            indexes[node] += 1
            previous_node = current_node
            previous_node_index = current_node_index
    
    for node in graph.nodes():
        
        # Add base nodes
        extended_graph.add_node(node)
        
        # Add connections between original node and all routes that contain that node
        for (route_node, data) in extended_graph.nodes(data=True):
            if data and data['original_node'] == node:
                
                # Add edge between original graph and route nodes
                extended_graph.add_edge(node, route_node, length=transfer_weight)

    return extended_graph


def calculate_transfers(path):
    """
    Calculates total number of transfers in a given path
    
    Args:
        path - a path, as a sequence of nodes.

    Returns: number of transfers (int)
    """
    
    # Return only base nodes (int values)
    base_nodes = [node for node in path if isinstance(node, int)]
    
    # Remove origin and destination
    transfers = len(base_nodes) - 2
    
    return transfers