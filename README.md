# Robust greedy algorithms for the Max-Cut relaxation
![Toy graph example B7](/imgs/B7_graph.png)

This is the accompanying code for the paper . In this work we propose an efficient, robust and greedy algorithm for the Max-Cut relaxation. Analysis of the algorithm demonstrate that it always converges to global optimum. The algorithm is robust to random and adversarial initial matrix solution and to random values $r$ of interpolation (see paper). Empirical results, which can be replicated with this implementation, supports our claims.

If you find this work useful please cite,

> here goes the reference to our paper

## Description
The input graphs must be stored in the `input` folder and must follow the same format as the G set graphs in this [link](https://sparse.tamu.edu/Gset).

We used an open source solver to find the optimal (relaxed) solution for the G set graphs and the toy size B graphs. The optimal solutions found are stored in the `solver_optimal_cuts.mat` in the `solver_solutions` folder.

To run the code:

``` python
python robust_maxcut_k2.py --input_graph G11 --fast_execution on
```

where:
- `input_graph` is the name of the graph to solve (G1, G2, ...)
- `max_iterations` is the maximum number of iterations for outer loop of the algorithm
- `tol` is the value of the tolerance used to evaluate convergence of algorithm
- `random_seed` is the value of the random seed for numpy (to replicate results)
- `fast_execution` is a value s.t., if True, some computations (like ranks and dual optimal) are avoided
- `rank` is the assumed rank of the graph

## Dependencies
- python 3.9.18
- numpy 1.26.0
- scipy 1.11.3
- matplotlib 3.5.3
- distutils 3.9.18

## License
Our implementation is open-sourced under the Apache-2.0 license. See the [LICENSE](https://github.com/b3r8/robust-greedy-maxcut/blob/main/LICENSE) file for details.

## Contact
Bernardo Gonzalez <[bernardo.gtorres@gmail.com](mailto:bernardo.gtorres@gmail.com)>
