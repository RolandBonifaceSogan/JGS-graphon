import numpy as np
from scipy.stats import rankdata
import ot
from scipy.ndimage import zoom

# =============================================================================
# GRAPHON FUNCTIONS
# =============================================================================

graphon_functions = {
    0: lambda x, y: x * y,
    1: lambda x, y: np.exp(-(x**0.7 + y**0.7)),
    2: lambda x, y: 0.25 * (x**2 + y**2 + np.sqrt(x) + np.sqrt(y)),
    3: lambda x, y: 0.5 * (x + y),
    4: lambda x, y: 1 / (1 + np.exp(-2 * (x**2 + y**2))),
    5: lambda x, y: 1 / (1 + np.exp(-(np.maximum(x, y) ** 2 + np.minimum(x, y) ** 4))),
    6: lambda x, y: np.exp(-(np.maximum(x, y) ** 0.75)),
    7: lambda x, y: np.exp(-0.5 * (np.minimum(x, y) + np.sqrt(x) + np.sqrt(y))),
    8: lambda x, y: np.log(1 +  0.5*np.maximum(x, y)),
    9: lambda x, y: np.abs(x - y),
    10: lambda x, y: 1 - np.abs(x - y),
    11: lambda x, y: 0.8 * ((x < 0.5) & (y < 0.5)),
    12: lambda x, y: 0.8 * (
        ((x < 0.5) & (y >= 0.5)) |
        ((x >= 0.5) & (y < 0.5))
    )
}
    
def generate_true_graphon_matrix(n, graphon_id):
    """Generate a probability matrix P(i,j)=W(U_i,U_j) using a regular grid of U."""
    U = np.linspace(1 / n, 1, n)
    if graphon_id in [1, 6, 7, 9, 10]:
        U = U[::-1]
    U_i, U_j = np.meshgrid(U, U, indexing="ij")
    true_graphon = graphon_functions[graphon_id]
    P = true_graphon(U_i, U_j)
    return P

def generate_graphs_from_graphon(vec_n, graphon_id):
    """Generate a collection of i.i.d. graphs from a chosen graphon function."""
    if graphon_id not in graphon_functions:
        raise ValueError(f"Invalid graphon_id. Choose from {list(graphon_functions.keys())}.")

    graphon_func = graphon_functions[graphon_id]
    graphs, positions = [], []

    for n in vec_n:
        U = np.random.uniform(0, 1, n)
        I, J = np.tril_indices(n, -1)
        P_lower = graphon_func(U[I], U[J])

        A = np.zeros((n, n), dtype=int)
        A[I, J] = np.random.binomial(1, P_lower)
        A += A.T  # symmetrize

        graphs.append(A)
        positions.append(U)

    return {"graphs": graphs, "positions": positions}

# =============================================================================
# METRICS
# =============================================================================

def graphon_L2_norm(W_est, W_true, ensure_monotonic=True):
    """Empirical L2 distance between two graphon matrices, ignoring diagonal entries."""
    if W_est.shape != W_true.shape:
        raise ValueError("W_est and W_true must have the same shape.")

    if ensure_monotonic:
        deg_est = np.mean(W_est, axis=1)
        W_est = W_est[np.argsort(deg_est)][:, np.argsort(deg_est)]
        deg_true = np.mean(W_true, axis=1)
        W_true = W_true[np.argsort(deg_true)][:, np.argsort(deg_true)]

    diff = W_est - W_true
    mask = ~np.eye(diff.shape[0], dtype=bool)
    return float(np.mean(diff[mask] ** 2))

def gw_distance(graphon, estimation):
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)

def resize_graphon_to_common(W, target_size):
    """Redimensionne une graphon à la taille cible avec interpolation."""
    if W.shape[0] == target_size:
        return W
    scale_factor = target_size / W.shape[0]
    return zoom(W, scale_factor, order=1)
