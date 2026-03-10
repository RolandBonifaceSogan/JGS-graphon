import numpy as np
from scipy.stats import rankdata
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import zoom


# =============================================================================
# JGS LATENT POSITION ESTIMATION
# =============================================================================

def estimate_latent_positions_from_all_graphs(graphs):
    """Estimate latent positions U_hat by sorting normalized degrees across all graphs."""
    all_degrees = np.concatenate([np.sum(graph, axis=1) / graph.shape[0] for graph in graphs])
    rank = rankdata(all_degrees, method="ordinal")
    N = len(all_degrees)
    U_hat_all = (rank - 0.5) / N

    U_hat_list, start = [], 0
    for graph in graphs:
        n_m = graph.shape[0]
        U_hat_list.append(U_hat_all[start:start + n_m])
        start += n_m
    return U_hat_list

# =============================================================================
# BLOCK PROCESSING FUNCTIONS (used by JGS estimator)
# =============================================================================

def _accumulate_edge_counts(graph, U_m, k, edge_frequencies, block_counts):
    """Accumulate edge counts and pair counts per (s,t) block for a single graph."""
    block_indices = np.floor(U_m * k).astype(int) + 1  # 1..k
    for s in range(1, k + 1):
        nodes_s = np.where(block_indices == s)[0]
        n_s = len(nodes_s)
        if n_s == 0:
            continue
        for t in range(s, k + 1):
            nodes_t = np.where(block_indices == t)[0]
            n_t = len(nodes_t)
            if n_t == 0:
                continue
            subgraph = graph[np.ix_(nodes_s, nodes_t)]
            edge_frequencies[s - 1, t - 1] += np.sum(subgraph)
            block_counts[s - 1, t - 1] += n_s * (n_s - 1) if s == t else n_s * n_t

def _finalize_block_matrices(edge_frequencies, block_counts):
    """Symmetrize, divide by counts, and fill empty blocks by simple neighbor averaging."""
    edge_frequencies[:] = edge_frequencies + edge_frequencies.T - np.diag(np.diag(edge_frequencies))
    block_counts[:] = block_counts + block_counts.T - np.diag(np.diag(block_counts))
    edge_frequencies[:] = edge_frequencies / np.maximum(block_counts, 1)

    k = edge_frequencies.shape[0]
    missing = (block_counts == 0)
    if np.any(missing):
        for s in range(k):
            for t in range(s, k):
                if missing[s, t]:
                    neighbors = []
                    if s > 0: neighbors.append(edge_frequencies[s - 1, t])
                    if s < k - 1: neighbors.append(edge_frequencies[s + 1, t])
                    if t > 0: neighbors.append(edge_frequencies[s, t - 1])
                    if t < k - 1: neighbors.append(edge_frequencies[s, t + 1])
                    if neighbors:
                        edge_frequencies[s, t] = np.mean(neighbors)
                        edge_frequencies[t, s] = edge_frequencies[s, t]
                        
                    
                    
def interpolate_graphon(W, target_size=1000):
    """Interpolate a graphon matrix to a desired resolution."""
    if W.shape[0] == target_size:
        return W
    scale_factor = target_size / W.shape[0]
    return zoom(W, scale_factor, order=1)

# =============================================================================
# JGS Graphon ESTIMATION
# =============================================================================


def joint_graph_sorting_estimate(graphs, k=None, U_hat_graphs=None, smoothing=False, target_size=None):
    """
    Estimate latent positions and the k x k edge-frequency matrix using Joint Graph Sorting (JGS).

    Parameters
    ----------
    graphs : list of np.ndarray
        List of adjacency matrices.
    k : int, optional
        Number of blocks. If None, it is selected automatically.
    U_hat_graphs : list of np.ndarray, optional
        Pre-computed latent positions.
    smoothing : bool, optional (default=False)
        If True, applies total variation smoothing to the estimated graphon.
    target_size : int, optional
        If provided, interpolates the estimated graphon to the desired resolution.
    
    Returns
    -------
    dict
        Contains:
        - "U_hat" : estimated latent positions per graph
        - "edge_frequencies" : estimated k x k graphon
        - "graphon_resized" : (optional) smoothed/interpolated graphon if options are enabled
    """
    M = len(graphs)

    # --- Automatic choice of k ---
    if k is None:
        vec_n = np.array([A.shape[0] for A in graphs])
        N = vec_n.sum()
        S = np.sum(vec_n ** 2)
        k = int(np.floor(min(S ** 0.25, N / (3.0*(M + np.log(N))))))

    # --- Estimate latent positions if not provided ---
    if U_hat_graphs is None:
        vec_n = np.array(list(map(len, graphs)))
        N = np.sum(vec_n)
        normalized_degrees = np.concatenate([np.sum(graph, axis=1) / n for graph, n in zip(graphs, vec_n)])
        U_hat = (rankdata(normalized_degrees, method="ordinal") - 0.5) / N
        ind = np.cumsum(np.insert(vec_n, 0, 0))
        U_hat_graphs = [U_hat[ind[m]:ind[m + 1]] for m in range(M)]

    # --- Initialize block matrices ---
    edge_frequencies = np.zeros((k, k), dtype=float)
    block_counts = np.zeros((k, k), dtype=float)

    # --- Accumulate counts from all graphs ---
    for graph, U_m in zip(graphs, U_hat_graphs):
        _accumulate_edge_counts(graph, U_m, k, edge_frequencies, block_counts)

    # --- Finalize the estimate ---
    _finalize_block_matrices(edge_frequencies, block_counts)

    # --- Optional smoothing ---
    if smoothing:
        h = 1 / k  # smoothing weight inversely proportional to resolution
        edge_frequencies = denoise_tv_chambolle(edge_frequencies, weight=h)

    # --- Optional interpolation to a target resolution ---
    graphon_resized = None
    if target_size is not None:
        graphon_resized = interpolate_graphon(edge_frequencies, target_size)

    return {
        "U_hat": U_hat_graphs,
        "edge_frequencies": edge_frequencies,
        "graphon_resized": graphon_resized
    }

