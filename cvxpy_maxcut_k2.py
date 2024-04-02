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
import cvxpy as cp
import matplotlib.pyplot as plt
from distutils.util import strtobool

def matrix_rank(A):
    '''
    Return rank of matrix using eigendecomposition method

    Input
    ----------
    A [numpy array] :           Input matrix

    Output
    -------
    rank_999 [int] :            Rank of input to keep 99.9% of "signal"
    eig_norm [numpy array] :    "Normalized" eigenvalues of matrix A
    '''

    # Compute eigenvalues of A (in our case, VV^t = X = A)
    eigenvalues, _ = np.linalg.eig(A)
    eigenvalues.sort()
    min_eig_val = min(eigenvalues)
    eigenvalues = eigenvalues[::-1]

    # L1-normalize (ignore negative eigenvalues)
    # (i.e., compute how much each eigenvalue contributes to "signal")
    if len(np.where(eigenvalues < 0)[0]) > 0:
        r = np.where(eigenvalues < 0)[0][0]
        eig_norm = eigenvalues[:r]/sum(eigenvalues[:r])
    else:
        eig_norm = eigenvalues/sum(eigenvalues)

    # Compute how many eigenvalues do we need to keep % of "signal"
    eigenvalues_norm_cum = np.cumsum(eig_norm)
    rank_999 = np.where(eigenvalues_norm_cum >= .999)[0][0]+1

    return rank_999, min_eig_val, eig_norm

def algorithm(graph, max_iterations, tol, W, n):
    '''
    Use CVXPY to solve Max-Cut relaxation for k=2.
    
    Inputs:
    ----------
    graph [str]:                      Input graph name (G1, G2, ...)
    max_iterations [int]:             Max iterations (outer loop)
    tol [float]:                      Tolerance for convergence
    W [scipy csc array]:              Sparse matrix of weights W
    n [int]:                          Number of nodes
    
    Output:
    ----------
    None (Solution, best cut, etc., is stored to disk)
    '''
    print('\nUsing CVXPY to solve Max-Cut problem\n')

    # PRELIMINARIES
    print('\nSetting preliminaries...\n')

    # Define variable to optimize
    X = cp.Variable((n,n), PSD=True)

    # Define constraints
    constraints = []
    for i in range(n):
        constraints.append(X[i,i]==1)

    # Compute constant part of function to optimize
    opt_constant = W.data.sum()

    # Define function to optimize
    obj = 0.25*(opt_constant - cp.trace(W @ X))

    # Define optimization problem
    prob = cp.Problem(cp.Maximize(obj), constraints)

    # See more info about solver at:
    # https://www.cvxpy.org/api_reference/cvxpy.problems.html
    # https://www.cvxpy.org/tutorial/advanced/index.html

    # START ALGORITHM
    prob.solve(verbose=True, eps_abs=tol, eps_rel=0, max_iters=max_iterations)
    # FINISH ALGORITHM

    # SOLUTION

    # Get final primal optimal
    optimal_cut = prob.value

    # Get solve time
    solve_time = prob.solver_stats.solve_time

    # Get number of iterations performed by solver
    iterations = prob.solver_stats.num_iters

    # Get final duality gap
    duality_gap = prob.solver_stats.extra_stats['info']['gap']

    # Get final dual residual
    dual_residual = prob.solver_stats.extra_stats['info']['res_dual']

    # Get final primal residual
    primal_residual = prob.solver_stats.extra_stats['info']['res_pri']

    # Compute final spectrum of solution matrix V
    rank999, min_eig_val, eigenvalues_norm = matrix_rank(X.value)

    print('\nNumber of iterations: {}'.format(iterations))
    print('Execution time [s]: {}'.format(solve_time))
    print('Optimal cut found: {}'.format(optimal_cut))
    print('Final duality gap: {}'.format(duality_gap))
    print('Final dual residual: {}'.format(dual_residual))
    print('Final primal residual: {}'.format(primal_residual))
    print('Final rank (99.9%) of V: {}'.format(rank999))
    print('Minimum eigenvalue: {}'.format(min_eig_val))

    # Sanity checks
    print('\nSanity check (this value should be larger or equal to -1):')
    print('min(X): {}'.format(np.min(X.value)))
    print('\nSanity check (this value should be smaller or equal to 1):')
    print('max(X): {}'.format(np.max(X.value)))

    # Store solution in memory
    to_file = {'k': 2, 'graph': graph, 'n': n, 'tolerance': tol,
               'X': X.value,  'optimal_cut': optimal_cut,
               'dual_optimal': optimal_cut+duality_gap,
               'dual_gap': duality_gap, 'dual_residual': dual_residual,
               'primal_residual': primal_residual, 'iterations': iterations,
               'final_rank': rank999, 'time': solve_time,
               'min_eig': min_eig_val}

    results_dir = './solver_solutions/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    scipy.io.savemat(results_dir+graph+'_k2_cvxpy.mat', to_file)
    print('\nSolution, optimal cut, and related info stored at {}'.
           format(results_dir))

    # Check if correctly saved
    from_file = scipy.io.loadmat(results_dir+graph+'_k2_cvxpy.mat')
    print('\nOptimal cut saved to file: {}'.
          format(from_file['optimal_cut'][0][0]))

    # Solution spectrum plot
    plots_dir = './solver_solutions/plots/'+graph+'_k2_cvxpy/'
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Plot final spectrum of solution
    x = range(rank999)
    fig, ax = plt.subplots()
    ax.plot(x, eigenvalues_norm[:rank999], 'ro-',
            label='spectrum (99.9%) of X', linewidth=1)
    ax.set(xlabel='i-th eigenvalue', ylabel='normalized eigenvalue',
           title='spectrum (99.9%) of V')

    ax.grid()
    plt.legend()
    plt.savefig(plots_dir+'final_spectrum_'+graph+'_k2_cvxpy.png')

    print('\nSolution spectrum plot stored at {}'.format(plots_dir))


def main():
    # For replicability
    np.random.seed(seed=args.random_seed)

    print('\nReading input...')

    # Read input
    input_dict = scipy.io.loadmat('./inputs/'+args.input_graph+'.mat')
    W = input_dict['Problem'][0][0][1]
    n = W.shape[0]
    m = W.size/2

    print('\nSolving Max-k-Cut relaxation for k=2, graph {}, using CVXPY'.
          format(args.input_graph))
    print('Max number of iterations: {}'.format(args.max_iterations))
    print('Convergence tolerance: {}'.format(args.tol))
    print('Number of nodes: {}'.format(n))
    print('Number of edges: {}'.format(m))

    # Sanity check: check max and min values in W
    print('max(W): {}'.format(max(W.data)))
    print('min(W): {}'.format(min(W.data)))

    # Algorithm
    algorithm(args.input_graph, args.max_iterations, args.tol, W, n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Greedy Max-Cut relaxation solver')
    parser.add_argument('--input_graph', type=str, default='G11',
                        metavar='Input graph', help='G1, G2, ...')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        metavar='Max iterations',
                        help='maximum number of iterations for outer loop')
    parser.add_argument('--tol', type=float, default=0.001,
                        metavar='Tolerance',
                        help='used to evaluate convergence of algorithm')
    parser.add_argument('--random_seed', type=int, default=0,
                        metavar='Random seed', help='random seed for numpy')

    args = parser. parse_args()

    main()
