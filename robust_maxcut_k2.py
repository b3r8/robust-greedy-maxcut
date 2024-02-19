"""
@author: Bernardo Gonzalez
@email: bernardo.gtorres@gmail.com
"""

from __future__ import print_function
import os
import time
import argparse

import scipy
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool

def matrix_rank(A):
    '''
    Return rank of matrix using SVD method

    Input
    ----------
    A [numpy array] :        Input matrix

    Output
    -------
    rank_99 [int] :             Rank of input to keep 99% of "signal"
    rank_999 [int] :            Rank of input to keep 99.9% of "signal"
    eig_norm [numpy array] :    "Normalized" eigenvalues of matrix A
    '''

    # Compute singular values of input
    singular_values = np.linalg.svd(A, compute_uv=False)

    # Compute eigenvalues of AA^t (in our case, VV^t = X)
    eigenvalues = singular_values**2

    # L1-normalize
    # (i.e., compute how much each eigenvalue contributes to "signal")
    eig_norm = eigenvalues/sum(eigenvalues)

    # Compute how many eigenvalues do we need to keep % of "signal"
    eigenvalues_norm_cum = np.cumsum(eig_norm)
    rank_99 = np.where(eigenvalues_norm_cum >= .99)[0][0]
    rank_999 = np.where(eigenvalues_norm_cum >= .999)[0][0]

    return rank_99, rank_999, eig_norm

def algorithm1(graph, max_iterations, tol, fast_execution, W, neighbors, n,
               solver_cut):
    '''
    Algorithm 1 from paper. Solve Max-Cut relaxation for k=2.
    
    Inputs:
    ----------
    graph [str]:                      Input graph name (G1, G2, ...)
    max_iterations [int]:             Max iterations (outer loop)
    tol [float]:                      Tolerance for convergence
    fast_execution [bool]:            If True, avoid extra computations
    W [scipy csc array]:              Sparse matrix of weights W
    neighbors [dictionary]:           Neighbors of every node
    n [int]:                          Number of nodes
    solver_cut [float]:               Optimal cut found by solver

    Output:
    ----------
    None (Solution, best cut, etc., is stored to disk)
    '''
    print('\nExecuting algorithm 1 from paper')

    # PRELIMINARIES

    # Array to store s values through execution (see paper)
    S = []

    # Array to store cut values through execution
    CUT = []

    # Array to store duality gap values through execution
    GAP = []

    # Array to store solution matrix (V) rank (99% preserv.)
    RANK99 = []

    # Array to store solution matrix (V) rank (99.9% preserv.)
    RANK999 = []

    # Array to store dual optimal values through execution
    DUAL_OPT = []

    # Array to store cos(theta) values through execution (see paper)
    COS_THETA = []

    # See below
    time_to_99 = -1

    # Inner step counter
    step_counter = 0

    # Flag to indicate if stop criteria was met
    stop_criteria = False

    # Initial solution (see paper)
    solution = np.eye(n,n)

    # Append initial cut to array
    brute_cut = 0
    for i in range(n):
        for j in neighbors[i]:
            w = W[(i,j)]
            brute_cut += w*(1. - np.dot(solution[i], solution[j]))

    cut = 0.25*brute_cut
    CUT.append(cut)

    # Compute constant part of function to optimize
    opt_constant = 0.25*W.data.sum()

    # Compute tolerance for angle (see paper)
    angle_tol = (tol**2)/2

    # START ALGORITHM
    # To measure execution time
    start_time = time.time()

    # Outer loop in algorithm 1 of paper (while loop in paper)
    for t in range(max_iterations):
        print('iteration number: {}'.format(t))

        # Update every vector greedily
        # (this loop is done in order in paper)
        for i in np.random.permutation(n):
            v = solution[i]

            # Compute partial derivative EFFICIENTLY
            # (only non-zero weights)
            partial_deriv = np.zeros(n)
            for j in neighbors[i]:
                partial_deriv -= W[(i,j)]*solution[j]

            # New vector is the partial derivative normalized
            # (i.e., norm = 1)
            s = np.linalg.norm(partial_deriv)
            S.append(s)

            # if disconnected node (probably)
            if s == 0.:
                new_v = v
            else:
                new_v = partial_deriv/s

            # Compute and store cos_theta
            cos_theta = np.dot(v, new_v)
            COS_THETA.append(cos_theta)

            # Compute and store new cut
            cut_improvement = 0.5*s*(1-cos_theta)    # See paper
            cut = CUT[-1]+cut_improvement
            CUT.append(cut)

            # Update V_i
            solution[i] = new_v

            # Update step counter
            step_counter += 1

            # Measure how long it took the algorithm to reach 99%
            # of commercial solver optimal solution
            if cut >= .99*solver_cut and time_to_99 < 0:
                time_to_99 = time.time() - start_time
                iter_to_99 = t
                step_to_99 = step_counter

        # Compute and store current dual optimal (see paper)
        DUAL_OPT.append((sum(S[0-n:])/4) + opt_constant)

        # Compute duality gap
        GAP.append(DUAL_OPT[-1] - CUT[-1-n])

        if not(fast_execution):
            # Compute and store rank of updated matrix V
            rank99, rank999, _ = matrix_rank(solution)
            RANK99.append(rank99)
            RANK999.append(rank999)

        # Stop criteria (All thetas=0 and strong duality, see paper)
        if min(COS_THETA[0-n:]) >= 1-angle_tol and GAP[-1] <= tol:
            print('Stop criteria reached')
            stop_criteria = True
            break

    # To measure execution time
    duration = time.time() - start_time

    # FINISH ALGORITHM

    # SOLUTION

    # Compute difference between our solution and solver solution
    difference = solver_cut-cut

    # Compute final spectrum of solution matrix V

    _, rank999, eigenvalues_norm = matrix_rank(solution)

    print('\nNumber of iterations: {}'.format(t))
    print('Stop criteria reached: {}'.format(stop_criteria))
    print('Final rank (99.9%) of V: {}'.format(rank999))
    print('Optimal cut found: {}'.format(CUT[-1]))
    print('Dual optimal found: {}'.format(DUAL_OPT[-1]))
    print('Final duality gap: {}'.format(GAP[-1]))
    print('Execution time [s]: {}'.format(duration))
    print('Iterations to reach 99% of solver solution: {}'.format(iter_to_99))
    print('Execution time to reach 99% of solver solution [s]: {}'.
          format(time_to_99))
    print('Difference between solver solution and our solution: {}'
              .format(difference))
    print('Difference between solver solution and our solution (as %): {} %'
              .format(difference*100/solver_cut))

    if n < 10000:
        # Compute X: matrix of dot products (Gram matrix)
        X = np.matmul(solution, solution.T)

        # Sanity checks
        print('\nSanity check (this value should be larger or equal to -1):')
        print('min(X): {}'.format(np.min(X)))

    # Recompute final cut
    # This value should be equal to last value stored in CUT array
    brute_cut = 0
    for i in range(n):
        for j in neighbors[i]:
            w = W[(i,j)]
            if n < 10000:
                brute_cut += w*(1. - X[i][j])
            else:
                brute_cut += w*(1. - np.dot(solution[i], solution[j]))

    cut = 0.25*brute_cut

    print('\nSanity check (these values should be equal):')
    print(cut)
    print(CUT[-1])

    # Store solution in memory
    to_file = {'k': 2, 'graph': graph, 'n': n, 'tolerance': tol,
               'V': solution,  'optimal_cut': cut,
               'dual_optimal': DUAL_OPT[-1], 'dual_gap': GAP[-1],
               'iterations': t, 'stop_criteria_reached': stop_criteria,
               'final_rank': rank999, 'time': duration,
               'steps_to_99': step_to_99, 'iterations_to_99': iter_to_99,
               'time_to_99': time_to_99,
               'difference_wrt_optimal_solver': difference,
               'diff_wrt_optimal_solver_as_perc': difference*100/solver_cut,
               'initial_rank': n}

    results_dir = './python_solutions/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    scipy.io.savemat(results_dir+graph+'_k2.mat', to_file)
    print('\nSolution, optimal cut, and related info stored at {}'.
           format(results_dir))

    # Check if correctly saved
    from_file = scipy.io.loadmat(results_dir+graph+'_k2.mat')
    print('Optimal cut saved to file: {}'.
          format(from_file['optimal_cut'][0][0]))

    # Plots of convergence
    plots_dir = './python_solutions/plots/'+graph+'_k2/'
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Plot cut convergence
    x = range(len(CUT))
    fig, ax = plt.subplots()
    ax.plot(x, CUT, label='primal solution')
    ax.set(xlabel='inner step', ylabel='optimal cut',
        title='cut convergence')

    # Add best solution found by solver
    ax.axhline(y = solver_cut, color = 'r',
            linestyle = 'dashed',
            label='solver solution')

    # Add step when we reach 99% of solver solution
    ax.axvline(x = step_to_99, color = 'g',
            linestyle = 'dashed',
            label='steps to reach 99% of solver solution')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'cut_convergence_'+graph+'_k2.png')

    # Plot s convergence
    x = range(len(S))
    fig, ax = plt.subplots()
    ax.plot(x, S, label='$s_i$: norm of derivative')
    ax.set(xlabel='inner step', ylabel='$s_i$',
        title='$s_i$ convergence')

    # Add step when we reach 99% of solver solution
    ax.axvline(x = step_to_99, color = 'g',
            linestyle = 'dashed',
            label='steps to reach 99% of solver solution')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'s_convergence_'+graph+'_k2.png')

    # Plot cos(theta) convergence
    x = range(len(COS_THETA))
    fig, ax = plt.subplots()
    ax.plot(x, COS_THETA, label='cos($\\theta_i$)')
    ax.set(xlabel='inner step',
           ylabel='cos($\\theta_i$)',
           title='cos($\\theta_i$) convergence')

    # Add step when we reach 99% of solver solution
    ax.axvline(x = step_to_99, color = 'g',
            linestyle = 'dashed',
            label='steps to reach 99% of solver solution')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'cos_theta_convergence_'+graph+'_k2.png')

    # Plot dual optimal convergence
    x = range(len(DUAL_OPT))
    fig, ax = plt.subplots()
    ax.plot(x, DUAL_OPT, label='dual solution')
    ax.set(xlabel='iteration', ylabel='dual optimal',
           title='dual convergence')

    # Add best solution found by solver
    ax.axhline(y = solver_cut, color = 'r',
               linestyle = 'dashed',
               label='solver solution')

    # Add step when we reach 99% of solver solution
    ax.axvline(x = iter_to_99, color = 'g',
               linestyle = 'dashed',
               label='iterations to reach 99% of solver solution')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'dual_convergence_'+graph+'_k2.png')

    # Plot duality gap convergence
    x = range(len(GAP))
    fig, ax = plt.subplots()
    ax.plot(x, GAP, label='duality gap')
    ax.set(xlabel='iteration', ylabel='gap',
           title='duality gap convergence')

    # Add step when we reach 99% of solver solution
    ax.axvline(x = iter_to_99, color = 'g',
               linestyle = 'dashed',
               label='iterations to reach 99% of solver solution')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'gap_convergence_'+graph+'_k2.png')

    # Plot final spectrum of solution
    x = range(rank999)
    fig, ax = plt.subplots()
    ax.plot(x, eigenvalues_norm[:rank999], 'ro-',
            label='spectrum (99.9%) of V', linewidth=1)
    ax.set(xlabel='i-th eigenvalue', ylabel='normalized eigenvalue',
           title='spectrum (99.9%) of V')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'final_spectrum_'+graph+'_k2.png')

    if not(fast_execution):
        # Plot solution rank convergence
        x99 = range(len(RANK99))
        fig, ax = plt.subplots()
        ax.plot(x99, RANK99, label='rank (99%) of V')

        x999 = range(len(RANK999))
        ax.plot(x999, RANK999, label='rank (99.9%) of V')
        ax.set(xlabel='iteration', ylabel='rank(V)',
            title='solution rank')

        # Add step when we reach 99% of solver solution
        ax.axvline(x = iter_to_99, color = 'g',
                linestyle = 'dashed',
                label='iterations to reach 99% of solver solution')

        ax.grid()
        plt.legend()
        plt.savefig(plots_dir+'rank_convergence_'+graph+'_k2.png')

    print('\nConvergence plots stored at {}'.format(plots_dir))


def main():
    print('\nSolving Max-k-Cut relaxation for k=2, graph {}'.
          format(args.input_graph))
    print('Max number of iterations: {}'.format(args.max_iterations))
    print('Convergence tolerance: {}'.format(args.tol))
    
    # For replicability
    np.random.seed(seed=args.random_seed)

    # Read input
    input_dict = scipy.io.loadmat('./inputs/'+args.input_graph+'.mat')
    W = input_dict['Problem'][0][0][1]
    n = W.shape[0]
    m = W.size/2
    print('number of nodes: {}'.format(n))
    print('number of edges: {}'.format(m))

    # For efficiency: create a dictionary of neighbors for every node
    neighbors = {}
    for i in range(n):
        neighbors[i] = np.nonzero(W[i].toarray()[0])[0]

    # Compute maximum and minimum degree of graph,
    # degree as number of neighbors, not sum of adjacent edge's weights
    maximum_degree = 0
    minimum_degree = n
    for i in neighbors:
        maximum_degree = max(len(neighbors[i]), maximum_degree)
        minimum_degree = min(len(neighbors[i]), minimum_degree)

    print('Maximum degree of G: {}'.format(maximum_degree))
    print('Minimum degree of G: {}'.format(minimum_degree))

    # Sanity check: check max and min values in W
    print('max(W): {}'.format(max(W.data)))
    print('min(W): {}'.format(min(W.data)))

    # Sanity check: check if disconnected graph
    isolated_nodes_counter = 0
    for i in range(n):
        if len(neighbors[i]) < 1:
            if isolated_nodes_counter == 0:
                print('\nDisconnected graph {}'.format(args.input_graph))

            print('Isolated node: {}'.format(i))
            isolated_nodes_counter += 1

    if isolated_nodes_counter > 0:
        print('Reminder: there are {} disconnected nodes in graph'.
              format(isolated_nodes_counter))
        print('This will affect the final rank computed by algorithm')

    # Read solution found by commercial solver
    solver = scipy.io.loadmat('./solver_solutions/solver_optimal_cuts.mat')
    if len(args.input_graph) == 2:
        key = args.input_graph[0]+'0'+args.input_graph[1]+'_k2'
    else:
        key = args.input_graph+'_k2'

    solver_cut = solver[key][0][0]
    print('Optimal value found by solver: {}'.format(solver_cut))

    # Algorithm
    algorithm1(args.input_graph, args.max_iterations, args.tol,
               args.fast_execution, W, neighbors, n, solver_cut)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Greedy Max-Cut relaxation solver')
    parser.add_argument('--input_graph', type=str, default='G11',
                        metavar='Input graph', help='G1, G2, ...')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        metavar='Max iterations',
                        help='maximum number of iterations for outer loop')
    parser.add_argument('--tol', type=float, default=0.00001,
                        metavar='Tolerance',
                        help='used to evaluate convergence of algorithm')
    parser.add_argument('--random_seed', type=int, default=0,
                        metavar='Random seed', help='random seed for numpy')
    parser.add_argument('--fast_execution', type=lambda x: bool(strtobool(x)),
                        default=False, metavar='Fast execution option',
                        help='If True, some computations (rank of current\
                        solution) are avoided')

    args = parser. parse_args()

    main()
