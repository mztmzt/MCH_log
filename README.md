# Matrix Completion with Hypergraphs: Sharp Thresholds and Efficient Algorithms

This repository contains the experimental code for our paper, *"Matrix Completion with Hypergraphs: Sharp Thresholds and Efficient Algorithms."* (Accepted by Learning on Graphs Conference(LoG) 2024). The code is organized to allow straightforward reproduction of the experiments presented in Figure 3 of the paper, including both synthetic and real-world datasets.

## Synthetic Data Experiments (Figure 3a, b, c)

To reproduce the experiments on synthetic datasets (subplots 3a, 3b, and 3c), run `synthetic_graph_experiments.py`. By adjusting the following parameters, you can obtain the results corresponding to each subplot:

- `n`: Number of users
- `m`: Number of items
- `gamma`: Minimum difference between rating vectors from distinct clusters
- `theta`: Random flipped noise probability
- `I1`, `I2`: Graph/Hypergraph quality

## Real-world Data Experiments (Figure 3d, e)

To reproduce the results for the real-world dataset experiments in Figure 3d, run `real_graph_experiments.py`. The results of Figure 3e can be obtained by adjusting the edge sampling parameter `q` within `real_graph_experiments.py`.


## Requirements

The following libraries are required to run the experiments:

- `networkx >= 2.5`
- `numpy >= 1.21`
- `scikit-learn >= 0.24`
- `matplotlib >= 3.4`
- `scipy >= 1.7`

Please ensure these packages are installed in your environment with the specified versions or later.
