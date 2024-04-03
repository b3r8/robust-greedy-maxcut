// algorithm2_snap_fixed_rank.cpp
// Bernardo Gonzalez
// bernardo.gtorres@gmail.com
// Algorithm 2 of paper (reduced rank)
// Modified for large graphs in SNAP dataset
// Due memory issues, rank p is fixed to 100

#include <math.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
 
int main(int argc, char** argv){
  /* Algorithm 2 from paper. Solve Max-Cut relaxation for k=2.
     Inputs:
     -------------------
     filename [file]:               Input file with edge list
     tolerance [double]:            Tolerance for convergence
     max_iters [int]:               Max. iterations (outer loop)

     Output:
     -------------------
     None (Solution, best cut, etc., is stored to disk)
*/

  // Check for arguments
  if (argc != 4){
    cout << "\nInput error" << endl;
    cout << "Usage: " << argv[0] << " filename tolerance max_iters"
    << endl;
    cout << "Example:" << argv[0] << " ./cpp_inputs/G11.txt 0.005 500"
    << endl;
    return -1;
  }

  int n,                        // Number of nodes/vectors (read from file)
      m,                        // Number of edges (read from file)
      t,                        // Iteration counter
      rank,                     // Initial rank of solution matrix (V)
      row_index,                // To read input file
      col_index,                // To read input file
      max_iterations,           // Max. number of iterations for algorithm
      inner_step_counter,       // Inner loop step counter
      stop_criteria = 0;        // Stop criteria flag

  double s,                     // Norm of partial derivative (see paper)
         tol,                   // Tolerance for convergence
         cut,                   // Current maximum cut found
         gap,                   // Current duality gap
         v_norm,                // Norm of vector v
         dual_opt,              // Current dual optimum
         prev_cut,              // To store previous cut
         cos_theta,             // Current cos(theta) (see paper)
         brute_cut,             // To compute initial and final cut
         average_s,             // To compute average s in an iteration
         exec_time,             // Used to compute execution time
         dot_product,           // To hold dot products of vectors
         opt_constant,          // Constant present in function to optimize
         dual_residual,         // To compute dual residual in an iteration
         cut_improvement,       // Improvement made by greedy step
         average_cos_theta;     // To compute average cos_theta

  string input_g,               // Input graph (G1, G2, ...)
         results_dir,           // Directory to store results
         results_file,          // Name of file to store optimum cut, etc.
         records_file;          // Name of file to store records (see below)

  clock_t start,                // Used to compute execution time
          end;                  // Used to compute execution time

  // IMPORTANT NOTE: Eigen library matrices are stored in column major by def.
  // Is important to keep this in mind for writing efficient code

  MatrixXd solution;            // Solution matrix (V)

  VectorXd v,                   // To hold current vector v_i to update
           new_v,               // new vector v_i (updated)
           partial_deriv;       // unnormalized new vector v_i (see paper)

  // Random seed
  srand (0);

  // Read input graph string, convergence tolerance and max. iterations

  input_g = argv[1];
  input_g = input_g.substr(strlen(argv[1])-7, 3);
  tol = atof(argv[2]);
  max_iterations = atoi(argv[3]);

  // If error while opening input file
  ifstream fin(argv[1]);
  if (!fin.is_open()){
    cout << "Failed to open " << argv[1] << endl;
    return -1;
  }

  cout << "\nReading input..." << endl;

  // n is the number of nodes, m the number of edges
  fin >> n >> m;

  // Initialize adjacency list (use for efficiency)
  vector< vector<int> > neighbors(n);

  // Initialize constant present in optimization function
  opt_constant = 0.0;

  // Set rank (see paper)
  rank = 100;

  // Read input graph/matrix (edge list)
  // Use a triplet list to read sparse matrix

  for (int i = 0; i < m; i++){
    fin >> row_index;
    fin >> col_index;
    opt_constant += 1.;
    neighbors[row_index].push_back(col_index);
    neighbors[col_index].push_back(row_index);
  }

  if (fin.fail() or fin.eof()){
    cout << "Error while reading " << argv[1] << endl;
    return -1;
  }

  fin.close();

  opt_constant *= 0.5;

  // Preliminaries
  cout << "\nSetting preliminaries..." << endl;

  // Records of algorithm (to analyze execution)
  double S[max_iterations],             // Array to store average s values
         CUT[1 + max_iterations],       // Array to store cut values
         GAP[max_iterations],           // Array to store gap values
         DUAL_OPT[max_iterations],      // Array to store dual opt. values
         DUAL_RESD[max_iterations],     // Array to store dual residual values
         COS_THETA[max_iterations];     // Array to store avg. cos(theta)

  // Initialize solution matrix V
  solution = MatrixXd::Random(rank,n);
  for (int i = 0; i < n; i++){
    v = solution.col(i);
    v_norm = v.norm();
    solution.col(i) = v/v_norm;
  }

  // Compute and append initial cut to array (and optim. constant)
  // Also, populate adjacency list
  brute_cut = 0.0;
  for (int i = 0; i < n; i++){
    for (int j: neighbors[i]){
      dot_product = solution.col(i).dot(solution.col(j));
      brute_cut += 1.0 - dot_product;
    }
  }

  prev_cut = 0.25*brute_cut;
  CUT[0] = prev_cut;

  cout << "\nSolving Max-k-Cut relaxation for k=2, graph "  << input_g << endl;
  cout << "Max. number of iterations: "  << max_iterations << endl;
  cout << "Convergence tolerance: "  << tol << endl;
  cout << "Number of nodes: "  << n << endl;
  cout << "Number of edges: "  << m << endl;
  cout << "max(W): 1" << endl;
  cout << "min(W): 1" << endl;

  // Initialize inner step counter
  inner_step_counter = 0;

  // Algorithm
  cout << "\nExecuting algorithm 2 from paper...\n" << endl;
  start = clock();

  // Outer loop in algorithm 1 of paper (while loop in paper)
  for (t = 0; t < max_iterations; t++){
    cout << "iteration number: " << t << endl;

    // Reset dual_opt sum
    dual_opt = 0.0;

    // Reset average s
    average_s = 0.0;

    // Reset average cos(theta)
    average_cos_theta = 0.0;

    // Reset dual residual (see paper)
    dual_residual = 0.0;

    // Update each vector greedily
    for (int i = 0; i < n; i++){
      v = solution.col(i);

      // Compute partial derivative efficiently (only neighbors)
      partial_deriv.setZero(rank);
      for (int j: neighbors[i]){
        partial_deriv -= solution.col(j);
      }

      // New vector is the partial derivative normalized (i.e., norm = 1)
      s = partial_deriv.norm();
      average_s += s/n;

      if (s == 0.0){
        new_v.setZero(rank);
      } else{
        new_v = partial_deriv/s;
      }

      // Compute and store cos_theta
      cos_theta = v.dot(new_v);
      average_cos_theta += cos_theta/n;

      // Compute and store new cut (see paper)
      cut_improvement = 0.5*s*(1.0 - cos_theta);
      cut = prev_cut + cut_improvement;
      prev_cut = cut;

      // Update dual optimal
      //dual_opt += s*cos_theta;
      dual_opt += s;

      // Update dual residual (for convergence)
      dual_residual = max(dual_residual, 0.5*sqrt(s*cut_improvement));

      // Update V_i
      solution.col(i) = new_v;

      // Update step counter
      inner_step_counter += 1;

    }

    // Store average s value in this iteration
    S[t] = average_s;

    // Store average cos(theta) value in this iteration
    COS_THETA[t] = average_cos_theta;

    // Store cut value at the end of iteration
    CUT[t+1] = cut;

    // Compute and store current dual optimal (see paper)
    dual_opt *= 0.25;
    dual_opt += opt_constant;
    DUAL_OPT[t] = dual_opt;

    // Compute and store current duality gap
    //gap = cut - dual_opt;
    gap = dual_opt - CUT[t];
    GAP[t] = gap;

    // Store dual residual of this iteration
    DUAL_RESD[t] = dual_residual;

    // Stop criteria (Dual residual and duality gap, see paper)
    if (dual_residual <= tol and gap <= tol){
      cout << "Stop criteria reached" << endl;
      stop_criteria = 1;
      break;
    }
  }

  end = clock();
  exec_time = (double)(end-start)/CLOCKS_PER_SEC;

  // If we reached the maximum number of iterations, correct value of t
  if (t == max_iterations){
    t = t - 1;
  }

  // Solution

  // Print results to screen
  cout << "\nStop criteria reached: " << stop_criteria << endl;
  cout << "Number of iterations: " << t+1 << endl;
  cout << "Execution time [s]: " << exec_time << endl;
  cout << "Optimal cut found: " << cut << endl;
  cout << "Dual optimal found: " << dual_opt << endl;
  cout << "Final duality gap: " << gap << endl;
  cout << "Final dual residual: " << dual_residual << endl;
  cout << "Final primal residual: 0 (by design)" << endl;

  // Recompute final cut (sanity check)
  brute_cut = 0.0;
  for (int i = 0; i < n; i++){
    for (int j: neighbors[i]){
      dot_product = solution.col(i).dot(solution.col(j));
      brute_cut += 1.0 - dot_product;
    }
  }

  cut = 0.25*brute_cut;
  cout << "\nSanity check (these values should be equal):" << endl;
  cout << cut << endl;
  cout << CUT[t+1] << endl;

  // Store output in memory

  // Define file names to store output 
  results_dir = "./cpp_solutions/";
  results_file = results_dir + input_g + "_results_reduced_rank.txt";
  records_file = results_dir + input_g + "_records_reduced_rank.txt";

  // Save number of iterations, execution time, etc to file
  ofstream file_res(results_file);
  if (file_res.is_open())
  {
    cout << "\nSaving number of iterations, execution time, etc to file: " +
    results_file << endl;

    file_res << "Stop criteria reached: " << stop_criteria << endl;
    file_res << "Number of iterations: " << t+1 << endl;
    file_res << "Execution time [s]: " << exec_time << endl;
    file_res << "Optimal cut found: " << cut << endl;
    file_res << "Dual optimal found: " << dual_opt << endl;
    file_res << "Final duality gap: " << gap << endl;
    file_res << "Final dual residual: " << dual_residual << endl;
    file_res << "Final primal residual: 0" << endl;
    file_res.close();
  }

  // Save records to file (S, CUT, GAP, etc.)
  ofstream file_rec(records_file);
  if (file_rec.is_open())
  {
    cout << "\nSaving records to file (S, CUT, GAP, etc.): " + records_file
    << endl;

    file_rec << "S: " << endl;
    for (int i = 0; i < t+1; i++){
      file_rec << S[i] << " ";
    }
    file_rec << endl;

    file_rec << "CUT: " << endl;
    for (int i = 0; i < t+2; i++){
      file_rec << CUT[i] << " ";
    }
    file_rec << endl;

    file_rec << "COS_THETA: " << endl;
    for (int i = 0; i < t+1; i++){
      file_rec << COS_THETA[i] << " ";
    }
    file_rec << endl;

    file_rec << "GAP: " << endl;
    for (int i = 0; i < t+1; i++){
      file_rec << GAP[i] << " ";
    }
    file_rec << endl;

    file_rec << "DUAL_OPT: " << endl;
    for (int i = 0; i < t+1; i++){
      file_rec << DUAL_OPT[i] << " ";
    }
    file_rec << endl;

    file_rec << "DUAL_RESD: " << endl;
    for (int i = 0; i < t+1; i++){
      file_rec << DUAL_RESD[i] << " ";
    }
    file_rec << endl;
    
    file_rec.close();
  }

  return 0;
}

