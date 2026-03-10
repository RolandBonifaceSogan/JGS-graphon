

# Joint Graph Sorting (JGS)

Official implementation of the paper:

**Low-Complexity and Consistent Graphon Estimation from Multiple Networks**  
Roland B. Sogan and Tabea Rebafka  
*AISTATS 2026*

This repository contains the Python implementation of the **Joint Graph Sorting (JGS)** estimator for graphon estimation from multiple networks.

JGS jointly aligns the nodes of several observed graphs, possibly of different sizes and without node correspondence across networks, in order to estimate a common underlying graphon.  
The method achieves high estimation accuracy while remaining computationally efficient, scaling nearly linearly with the total number of observed edges.


## Requirements

Before running the scripts, please ensure that the following Python packages are installed:

```bash
pip install numpy scipy matplotlib POT scikit-image
```

You can also install all dependencies at once by running:

```bash
pip install -r requirements.txt
```

##  Usage

### Basic Example

```python
import numpy as np
import matplotlib.pyplot as plt
from src.jgs_utils import *
from src.jgs_estimator import *
```

### Simulation setup 

```python
vec_n = [100, 150, 200]      # sizes of observed graphs
graphon_id = 0               # choose from 0 to 11
graphon_size = 1000
```
### Data generation

```python
data = generate_graphs_from_graphon(vec_n, graphon_id)
graphs = data["graphs"]
```
### True graphon 
```python
W_true = generate_true_graphon_matrix(graphon_size, graphon_id)
```

### Graphon estimation
```python
est1 = joint_graph_sorting_estimate(graphs, smoothing=False, target_size=graphon_size)
est2 = joint_graph_sorting_estimate(graphs, smoothing=True, target_size=graphon_size)
```

### Evaluation 
```python
mse1 = graphon_L2_norm(est1["graphon_resized"], W_true)
mse2 = graphon_L2_norm(est2["graphon_resized"], W_true)
print(f"L2 error (no smoothing): {mse1:.4f}")
print(f"L2 error (with smoothing): {mse2:.4f}")
```

### Estimation with specific number of blocks

```python
est3 = joint_graph_sorting_estimate(graphs, k=50, smoothing=True, target_size=graphon_size)
mse3 = graphon_L2_norm(est3["graphon_resized"], W_true)
print(f"L2 error with k=50: {mse3:.4f}")
```
### Visualization 

```python
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(W_true, cmap='viridis')
plt.title('True Graphon')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(est1["edge_frequencies"], cmap='viridis')
plt.title('Estimated Graphon\n(No Smoothing)')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(est1["graphon_resized"], cmap='viridis')
plt.title('Resized Graphon\n(No Smoothing)')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(est2["edge_frequencies"], cmap='viridis')
plt.title('Estimated Graphon\n(With Smoothing)')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.imshow(est2["graphon_resized"], cmap='viridis')
plt.title('Resized Graphon\n(With Smoothing)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Notes on Reproducibility

The code in this repository reproduces the main experiments for the Joint Graph Sorting (JGS) estimator.  
For full reproducibility of the comparative benchmarks (e.g., with **G-Mixup**, **SGWB**, **SIGL**, etc.), please ensure that:

1. You have downloaded or cloned the official implementations of these methods.  
2. All dependencies required by each method are installed (see their respective `requirements.txt` files).  
3. You execute the experiments in a compatible environment


## Citation

If you use this code or the JGS estimator in your work, please cite:

```bibtex
@inproceedings{sogan2026jgs,
  title={Low-Complexity and Consistent Graphon Estimation from Multiple Networks},
  author={Sogan, Roland B. and Rebafka, Tabea},
  booktitle={Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```



## License

This project is released under the MIT License. See the `LICENSE` file for details.
