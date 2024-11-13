import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize

def compute_transition_matrix(current_state, velocity, n_neighbors=500, eps=1e-5, sigma=5.0):
    n_cells = current_state.shape[0]

    if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=1e10, neginf=-1e10)
    
    zero_vel_rows = np.where(np.all(velocity == 0, axis=1))[0]
    if len(zero_vel_rows) > 0:
        velocity[zero_vel_rows] = eps

    state_distances = squareform(pdist(current_state, metric='euclidean'))
    neighbors_indices = np.argsort(state_distances, axis=1)[:, 1:n_neighbors + 1]
    transition_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        current_velocity = velocity[i]
        neighbor_positions = current_state[neighbors_indices[i]]
        direction_vectors = neighbor_positions - current_state[i]
        
        direction_vectors_normalized = normalize(direction_vectors, axis=1)
        current_velocity_normalized = normalize(current_velocity.reshape(1, -1), axis=1).flatten()
        cosine_similarities = np.dot(direction_vectors_normalized, current_velocity_normalized)
    
        distances_to_neighbors = state_distances[i, neighbors_indices[i]]
        distance_kernel = np.exp(-np.square(distances_to_neighbors) / (2 * sigma ** 2))
        
        transition_probs = cosine_similarities * distance_kernel
        transition_probs = np.clip(transition_probs, 0, 1 - eps) 

        transition_probs /= (transition_probs.sum() + eps)
        transition_matrix[i, neighbors_indices[i]] = transition_probs

    return transition_matrix

def diffusion_pseudotime(transition_matrix, start_cell=None):
    graph = csr_matrix(transition_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=True, return_labels=True)
    if start_cell is None:
        start_cell = np.argmax(transition_matrix.sum(axis=1))
        print(f"Automatically selected start cell: {start_cell}")
    else:
        print(f"Using specified start cell: {start_cell}")
        
    distances, predecessors = dijkstra(csgraph=graph, directed=True, 
                                     indices=start_cell, 
                                     return_predecessors=True)

    unreachable = np.isinf(distances)
    
    if unreachable.sum() > 0:
        max_finite_distance = np.max(distances[~np.isinf(distances)])
        distances[unreachable] = max_finite_distance * 1.1  
    
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    return normalized_distances


# import numpy as np
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import spearmanr

# def compute_transition_matrix(current_state, velocity, n_neighbors=50, eps=1e-5, sigma=1.0):
#     n_cells = current_state.shape[0]
    
#     if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
#         velocity = np.nan_to_num(velocity, nan=0.0, posinf=1e10, neginf=-1e10)
    
#     zero_vel_rows = np.where(np.all(velocity == 0, axis=1))[0]
#     if len(zero_vel_rows) > 0:
#         velocity[zero_vel_rows] = eps

#     vel_similarities = 1 - squareform(pdist(velocity, metric='cosine'))
#     vel_similarities = np.clip(vel_similarities, 0, 1 - eps)
#     state_distances = squareform(pdist(current_state, metric='euclidean'))
#     distance_kernel = np.exp(-np.square(state_distances) / (2 * sigma ** 2))
#     transition_probs = vel_similarities * distance_kernel

#     for i in range(n_cells):
#         indices = np.argsort(transition_probs[i])[::-1]
#         transition_probs[i, indices[n_neighbors:]] = 0
        
#     zero_rows = np.where(np.sum(transition_probs, axis=1) == 0)[0]
#     for row in zero_rows:
#         transition_probs[row, np.argmax(vel_similarities[row])] = eps
    
#     row_sums = transition_probs.sum(axis=1)
#     transition_matrix = transition_probs / row_sums[:, np.newaxis]
    
    
#     return transition_matrix

# def compute_pseudotime(transition_matrix, start_cell=None, eps=1e-10):
#     n_cells = transition_matrix.shape[0]
#     if start_cell is None:
#         start_cell = np.argmax(transition_matrix.sum(axis=1))
#         print(f"Automatically selected start cell: {start_cell}")
#     else:
#         print(f"Using specified start cell: {start_cell}")
#     pseudotime = np.zeros(n_cells)
#     current_cell = start_cell
#     current_time = 0

#     visited = set([start_cell])
#     while len(visited) < n_cells:
#         probs = transition_matrix[current_cell]
#         next_cell = np.argmax(probs)
#         if next_cell in visited:
#             unvisited = list(set(range(n_cells)) - visited)
#             next_cell = unvisited[np.argmax(probs[unvisited])]
        
#         time_step = 1 / (probs[next_cell] + eps)
#         time_step = min(time_step, 1000)
#         current_time += time_step
#         pseudotime[next_cell] = current_time
#         visited.add(next_cell)
#         current_cell = next_cell

#     pseudotime = (pseudotime - np.min(pseudotime)) / (np.max(pseudotime) - np.min(pseudotime))
    
#     return pseudotime

