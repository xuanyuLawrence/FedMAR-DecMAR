import numpy as np
import networkx as nx
import random


import numpy as np
import networkx as nx
import random

def generate_adj_matrix_max(num_clients, max_connectivity, uniform_neighbors=False, one_max_connectivity=False, min_connectivity=2):
    """
    Generate an adjacency matrix for a connected decentralized network.

    Args:
        num_clients (int): Number of nodes in the network.
        max_connectivity (int): Desired maximum connectivity.
        uniform_neighbors (bool): All nodes have max_connectivity neighbors.
        one_max_connectivity (bool): Only one node has max_connectivity; others have min_connectivity.
        min_connectivity (int): Minimum number of neighbors for other nodes if one_max_connectivity is True.

    Returns:
        np.ndarray: Adjacency matrix A.
    """
    if num_clients < 2:
        raise ValueError("The network must have at least 2 clients.")
    if max_connectivity < 2 or min_connectivity < 1:
        raise ValueError("Connectivity values must be >= 1.")

    while True:
        G = nx.Graph()
        G.add_nodes_from(range(num_clients))

        # Ensure connectivity using a spanning tree
        for i in range(1, num_clients):
            parent = random.randint(0, i - 1)
            G.add_edge(parent, i)

        if one_max_connectivity:
            max_node = random.randint(0, num_clients - 1)
            for node in range(num_clients):
                desired = max_connectivity if node == max_node else min_connectivity
                while len(list(G.neighbors(node))) < desired:
                    potential = list(set(range(num_clients)) - {node} - set(G.neighbors(node)))
                    if not potential:
                        break  # no more candidates
                    neighbor = random.choice(potential)
                    G.add_edge(node, neighbor)
        else:
            for node in range(num_clients):
                desired = max_connectivity if uniform_neighbors else random.randint(2, max_connectivity)
                while len(list(G.neighbors(node))) < desired:
                    potential = list(set(range(num_clients)) - {node} - set(G.neighbors(node)))
                    if not potential:
                        break
                    neighbor = random.choice(potential)
                    G.add_edge(node, neighbor)

        if nx.is_connected(G):
            break

    return nx.to_numpy_array(G)

def generate_adj_matrix_avg(num_clients, avg_connectivity, uniform_neighbors=False):
    """
    Generate an adjacency matrix for a random connected network with a given average connectivity.

    Args:
        num_clients (int): Number of nodes in the network.
        avg_connectivity (float): Desired average number of neighbors per node.
        uniform_neighbors (bool): If True, all clients have the same number of neighbors.

    Returns:
        np.ndarray: Adjacency matrix A.
    """
    if num_clients < 2:
        raise ValueError("The network must have at least 2 clients.")
    if avg_connectivity < 1 or avg_connectivity > num_clients - 1:
        raise ValueError("Average connectivity must be between 1 and num_clients - 1.")

    total_edges = int((avg_connectivity * num_clients) // 2)

    while True:
        G = nx.Graph()
        G.add_nodes_from(range(num_clients))

        # Start with a spanning tree
        for i in range(1, num_clients):
            parent = random.randint(0, i - 1)
            G.add_edge(parent, i)

        current_edges = G.number_of_edges()
        remaining_edges = total_edges - current_edges

        # Add remaining edges
        if uniform_neighbors:
            target_degree = int(avg_connectivity)
            for node in range(num_clients):
                while len(list(G.neighbors(node))) < target_degree:
                    candidates = list(set(range(num_clients)) - {node} - set(G.neighbors(node)))
                    if not candidates:
                        break
                    neighbor = random.choice(candidates)
                    G.add_edge(node, neighbor)
        else:
            while G.number_of_edges() < total_edges:
                a, b = random.sample(range(num_clients), 2)
                if not G.has_edge(a, b):
                    if len(G[a]) < num_clients - 1 and len(G[b]) < num_clients - 1:
                        G.add_edge(a, b)

        if nx.is_connected(G):
            break

    return nx.to_numpy_array(G)

def save_adj_matrix(matrix, file_path):
    """
    Save the adjacency matrix to a file.

    Args:
        matrix (np.ndarray): The adjacency matrix to save.
        file_path (str): The file path to save the matrix.
    """
    np.save(file_path, matrix)

def load_adj_matrix(file_path):
    """
    Load the adjacency matrix from a file.

    Args:
        file_path (str): The file path to load the matrix from.

    Returns:
        np.ndarray: The loaded adjacency matrix.
    """
    return np.load(file_path)

def verify_adj_matrix_max(matrix, num_clients, max_connectivity):
    """
    Verify that the loaded adjacency matrix matches the given parameters.

    Args:
        matrix (np.ndarray): The adjacency matrix to verify.
        num_clients (int): The expected number of clients (nodes).
        max_connectivity (int): The maximum number of neighbors for each client.

    Returns:
        bool: True if the matrix is valid, False otherwise.
    """
    # Check that the matrix is square and has the correct size
    if matrix.shape != (num_clients, num_clients):
        return False
    
    # Check the connectivity constraints
    for i in range(num_clients):
        # Count the number of neighbors (non-zero entries in the row)
        num_neighbors = np.sum(matrix[i])
        if num_neighbors > max_connectivity:
            return False
    
    # Check if the matrix is symmetric (since it's undirected)
    if not np.allclose(matrix, matrix.T):
        return False

    # If all checks pass, return True
    return True

def verify_adj_matrix_avg(matrix, num_clients, avg_connectivity, tolerance=0.1):
    """
    Verify that the adjacency matrix meets the average connectivity criteria.

    Args:
        matrix (np.ndarray): The adjacency matrix to verify.
        num_clients (int): The expected number of nodes.
        avg_connectivity (float): The target average connectivity.
        tolerance (float): Acceptable relative deviation (e.g., 0.1 for ±10%).

    Returns:
        bool: True if the matrix is valid, False otherwise.
    """
    # Check shape
    if matrix.shape != (num_clients, num_clients):
        return False

    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        return False

    # Degree check
    degrees = matrix.sum(axis=1)
    avg_deg_actual = np.mean(degrees)

    lower_bound = avg_connectivity * (1 - tolerance)
    upper_bound = avg_connectivity * (1 + tolerance)

    if not (lower_bound <= avg_deg_actual <= upper_bound):
        return False

    return True

# Example: Get neighbors of a specific node
def get_neighbors(matrix, node_id):
    return [i for i, is_neighbor in enumerate(matrix[node_id]) if is_neighbor == 1]

