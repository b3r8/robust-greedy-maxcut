# Fast and Provable Greedy Algorithms for the Max-Cut Relaxation
![Toy graph example B7](/imgs/B7_graph.png)

This is the accompanying code for the paper "Fast and Provable Greedy Algorithms for the Max-Cut Relaxation". In this work we propose simple and provable, fast greedy algorithms to solve the semidefinite programming relaxation proposed by Goemans and Williamson.

Analysis of the algorithm demonstrate that it always converges to the global optimum. Cuts found by the algorithms are monotonically non-decreasing, and will keep improving until no improvement can be made. The C++ implementation
of algorithm 2 (low rank) can solve the Max-Cut relaxation for graphs with thousands of nodes in a matter of seconds in a desktop machine. Empirical results, which can be replicated with this implementation, supports our claims.

If you find this work useful please cite,

> here goes the reference to our paper

## Description

### C++ implementation
For the G set graphs, the input graphs must be stored in the `input` folder and must follow the same format as the example shown for G11 (`G11.txt` file) in this repo. The `G11.mat` file is the input file for the python implementations of the algorithm.

For the SNAP set graphs, the input graphs must be stored in the `input` folder and must follow the same format as the example shown for the Amazon graph (`amazon.txt` file) in this repo. The main difference with respect to the input files for the G set graphs is that the input files for the SNAP set graphs does not include the weights of the edges, because the weight is equal to one for all edges. This allow us to save space for storing these graphs.

Four algorithms were implemented in C++:
- `algorithm1.cpp`, implementation of Algorithm 1 (full rank) in the paper.
- `algorithm2.cpp`, implementation of Algorithm 2 (low rank) in the paper.
- `algorithm2_snap.cpp`, implementation of Algorithm 2 (low rank) in the paper, modified to read input graphs from the SNAP set (see previous paragraph).
- `algorithm2_snap_fixed_rank.cpp`, implementation of Algorithm 2 (low rank) in the paper, modified to read input graphs from the SNAP set (see previous paragraph) and modified to set the value of p = 100 (initial rank of solution matrix) to handle graphs with more than 1 million of nodes (see paper).

To compile algorithm 1:

``` cpp
g++ -I /path/to/folder/eigen-3.3.9/ algorithm1.cpp -o algorithm1 -O2
```

To run algorithm 1:

``` cpp
./algorithm1 ./inputs/G11.txt 0.001 1000
```

where:
- `G11.txt` is the name of the graph to solve (G1, G2, ...)
- `0.001` is the value of the tolerance used to evaluate convergence of algorithm
- `1000` is the maximum number of iterations for outer loop of the algorithm

The instructions to compile and run the rest of C++ implementations (algorithm2, algorithm2_snap, and algorithm2_snap_fixed_rank) are the same as for algorithm1 but with the name of the algorithm modified accordingly.

**Very important: do not forget to use the flag** `-O2` **to compile the codes, otherwise the implementations will be slow.**

### Python implementation
In addition to our implementation in C++, we implemented our algorithm in Python. This implementation is not as efficient in time and space as the implementation in C++, but allow us to easily plot the convergence of different quantities during the execution of the algorithm.

The input graphs must be stored in the `input` folder and must follow the same format as the G set graphs in this [link](https://sparse.tamu.edu/Gset). An example is shown for G11 (`G11.mat` file) in this repo.

We used an open source solver ([SCS](https://www.cvxgrp.org/scs/index.html) with [CVXPY](https://www.cvxpy.org/)) to find the optimal solution for the G set graphs to compare our results.

To run algorithm 1:

``` python
python robust_maxcut_k2.py --input_graph G11 --max_iterations 1000 --tol 0.001 --random_seed 0 --fast_execution on
```

To run algorithm 2 (low rank):

``` python
python robust_maxcut_k2_reduced_rank.py --input_graph G11 --max_iterations 1000 --tol 0.001 --random_seed 0 --fast_execution on
```

To run SCS (with CVXPY):

``` python
python cvxpy_maxcut_k2.py --input_graph G11 --max_iterations 1000 --tol 0.001 --random_seed 0
```

where:
- `input_graph` is the name of the graph to solve (G1, G2, ...)
- `max_iterations` is the maximum number of iterations for outer loop of the algorithm
- `tol` is the value of the tolerance used to evaluate convergence of algorithm
- `random_seed` is the value of the random seed for numpy (to replicate results)
- `fast_execution` is a value s.t., if True, some computations (solution matrix rank per iteration) are avoided (only available for algorithms 1 and 2, not for SCS)

## Dependencies

### C++
- [Eigen](https://libeigen.gitlab.io/docs/index.html) 3.3.9

### Python
- python 3.9.18
- numpy 1.26.0
- scipy 1.11.3
- matplotlib 3.5.3
- distutils 3.9.18
- cvxpy 1.4.2 (only used for comparison, not necessary for algorithms' implementation)

## Colab
You can also replicate our python implementation results with the following google colab notebooks:
- [Main example](https://colab.research.google.com/drive/1vZtJUD_Afd0HHdPcAthm5QdSYYSfaCJi?usp=sharing)

## License
Our implementation is open-sourced under the Apache-2.0 license. See the [LICENSE](https://github.com/b3r8/robust-greedy-maxcut/blob/main/LICENSE) file for details.

## Contact
Bernardo Gonzalez <[bernardo.gtorres@gmail.com](mailto:bernardo.gtorres@gmail.com)>
