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
    A [numpy array] :           Input matrix

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
    rank_99 = np.where(eigenvalues_norm_cum >= .99)[0][0]+1
    rank_999 = np.where(eigenvalues_norm_cum >= .999)[0][0]+1

    return rank_99, rank_999, eig_norm

def algorithm1(graph, max_iterations, tol, fast_execution, W, neighbors, n):
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

    Output:
    ----------
    None (Solution, best cut, etc., is stored to disk)
    '''
    print('\nSetting preliminaries...')

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

    # Array to store dual residual made per iteration
    # (i.e., 0.5*max improvement, see paper)
    DUAL_RESD = []

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

    # START ALGORITHM
    print('\nExecuting algorithm 1 from paper...\n')

    # To measure execution time
    start_time = time.time()

    # Outer loop in algorithm 1 of paper (while loop in paper)
    for t in range(max_iterations):
        print('iteration number: {}'.format(t))

        # To compute dual optimal
        dual_opt = 0.

        # To compute dual residual per iteration (for convergence)
        dual_residual = 0.

        # Update every vector greedily
        for i in range(n):
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
                new_v = np.zeros(n)
            else:
                new_v = partial_deriv/s

            # Compute and store cos_theta
            cos_theta = np.dot(v, new_v)
            COS_THETA.append(cos_theta)

            # Compute and store new cut (see paper)
            cut_improvement = 0.5*s*(1-cos_theta)
            cut = CUT[-1]+cut_improvement
            CUT.append(cut)

            # Update dual optimal
            dual_opt += s

            # Update max improvement (for convergence)
            dual_residual_candidate = 0.5*np.sqrt(s*cut_improvement)
            dual_residual = max(dual_residual, dual_residual_candidate)

            # Update V_i
            solution[i] = new_v

            # Update step counter
            step_counter += 1

        # Compute and store current dual optimal (see paper)
        #DUAL_OPT.append((sum(S[0-n:])/4.) + opt_constant)
        DUAL_OPT.append((0.25*dual_opt) + opt_constant)

        # Compute and store duality gap
        #GAP.append(CUT[-1] - DUAL_OPT[-1])
        GAP.append(DUAL_OPT[-1] - CUT[-1-n])


        # Store dual residual of this iteration
        DUAL_RESD.append(dual_residual)

        if not(fast_execution):
            # Compute and store rank of updated matrix V
            rank99, rank999, _ = matrix_rank(solution)
            RANK99.append(rank99)
            RANK999.append(rank999)

        # Stop criteria (Dual residual and dual gap, see paper)
        if DUAL_RESD[-1] <= tol and GAP[-1] <= tol:
            print('Stop criteria reached')
            stop_criteria = True
            break

    # To measure execution time
    duration = time.time() - start_time

    # FINISH ALGORITHM

    # SOLUTION

    # Compute final spectrum of solution matrix V
    _, rank999, eigenvalues_norm = matrix_rank(solution)

    print('\nStop criteria reached: {}'.format(stop_criteria))
    print('Number of iterations: {}'.format(t+1))
    print('Execution time [s]: {}'.format(duration))
    print('Optimal cut found: {}'.format(CUT[-1]))
    print('Dual optimal found: {}'.format(DUAL_OPT[-1]))
    print('Final duality gap: {}'.format(GAP[-1]))
    print('Final dual residual: {}'.format(DUAL_RESD[-1]))
    print('Final primal residual: 0 (by design)')
    print('Final rank (99.9%) of V: {}'.format(rank999))

    if n < 10000:
        # Compute X: matrix of dot products (Gram matrix)
        X = np.matmul(solution, solution.T)

        # Sanity checks
        print('\nSanity check (this value should be larger or equal to -1):')
        print('min(X): {}'.format(np.min(X)))
        print('\nSanity check (this value should be smaller or equal to 1):')
        print('max(X): {}'.format(np.max(X)))

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
               'V': solution,  'optimal_cut': CUT[-1],
               'dual_optimal': DUAL_OPT[-1], 'dual_gap': GAP[-1],
               'dual_residual': DUAL_RESD[-1], 'iterations': t+1,
               'stop_criteria_reached': stop_criteria, 'final_rank': rank999,
               'time': duration, 'initial_rank': n}

    results_dir = './python_solutions/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    scipy.io.savemat(results_dir+graph+'_k2.mat', to_file)
    print('\nSolution, optimal cut, and related info stored at {}'.
           format(results_dir))

    # Check if correctly saved
    from_file = scipy.io.loadmat(results_dir+graph+'_k2.mat')
    print('\nOptimal cut saved to file: {}'.
          format(from_file['optimal_cut'][0][0]))

    # Plots of convergence
    plt.rcParams['font.size'] = 21

    plots_dir = './python_solutions/plots/'+graph+'_k2/'
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Plot solution convergence
    x = range(len(CUT))
    fig, ax = plt.subplots()
    ax.plot(x, CUT, label='solution')
    ax.set(xlabel='inner step')

    ax.grid()
    plt.legend(loc='lower right')
    plt.savefig(plots_dir+'cut_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot s convergence
    x = range(len(S))
    fig, ax = plt.subplots()
    ax.plot(x, S, label='$s_i$')
    ax.set(xlabel='inner step')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'s_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot cos(theta) convergence
    x = range(len(COS_THETA))
    fig, ax = plt.subplots()
    ax.plot(x, COS_THETA, label='cos($\\theta_i$)')
    ax.set(xlabel='inner step')

    ax.grid()
    plt.legend(loc='lower right')
    plt.savefig(plots_dir+'cos_theta_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot dual optimal convergence
    x = range(len(DUAL_OPT))
    fig, ax = plt.subplots()
    ax.plot(x, DUAL_OPT, label='dual solution')
    ax.set(xlabel='iteration')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'dual_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot duality gap convergence
    x = range(len(GAP))
    fig, ax = plt.subplots()
    ax.plot(x, GAP, label='duality gap')
    ax.set(xlabel='iteration')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'gap_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot dual residual convergence
    x = range(len(DUAL_RESD))
    fig, ax = plt.subplots()
    ax.plot(x, DUAL_RESD, label='dual residual')
    ax.set(xlabel='iteration')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'dual_residual_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    # Plot final spectrum of solution
    x = range(rank999)
    fig, ax = plt.subplots()
    ax.plot(x, eigenvalues_norm[:rank999], 'ro-', linewidth=1)
    ax.set(xlabel='i-th eigenvalue', ylabel='normalized eigenvalue',
            title='spectrum (99.9%)')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'final_spectrum_'+graph+'_k2.png',
                bbox_inches='tight')

    if not(fast_execution):
        # Plot solution rank convergence
        x99 = range(len(RANK99))
        fig, ax = plt.subplots()
        ax.plot(x99, RANK99, label='rank (99%)')

        x999 = range(len(RANK999))
        ax.plot(x999, RANK999, label='rank (99.9%)')
        ax.set(xlabel='iteration')

        ax.grid()
        plt.legend()
        plt.savefig(plots_dir+'rank_convergence_'+graph+'_k2.png',
                bbox_inches='tight')

    print('\nConvergence plots stored at {}'.format(plots_dir))


def main():
    # For replicability
    np.random.seed(seed=args.random_seed)

    print('\nReading input...')

    # Read input
    input_dict = scipy.io.loadmat('./inputs/'+args.input_graph+'.mat')
    W = input_dict['Problem'][0][0][1]
    n = W.shape[0]
    m = 0.5*W.size

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

    print('\nSolving Max-k-Cut relaxation for k=2, graph {}'.
          format(args.input_graph))
    print('Max number of iterations: {}'.format(args.max_iterations))
    print('Convergence tolerance: {}'.format(args.tol))
    print('Number of nodes: {}'.format(n))
    print('Number of edges: {}'.format(m))
    print('max(W): {}'.format(max(W.data)))
    print('min(W): {}'.format(min(W.data)))
    print('Maximum degree of G: {}'.format(maximum_degree))
    print('Minimum degree of G: {}'.format(minimum_degree))

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

    # Algorithm
    algorithm1(args.input_graph, args.max_iterations, args.tol,
               args.fast_execution, W, neighbors, n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Greedy Max-Cut relaxation solver')
    parser.add_argument('--input_graph', type=str, default='G11',
                        metavar='Input graph', help='G1, G2, ...')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        metavar='Max iterations',
                        help='maximum number of iterations for outer loop')
    parser.add_argument('--tol', type=float, default=0.001,
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
