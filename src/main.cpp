#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <random>

#include "sgmres.hpp"
#include "dqgmres.hpp"
#include "utils.hpp"
#include "sketching.hpp"
#include "preconditioner.hpp"
#include <composyx.hpp>
#include <composyx/linalg/MatrixNorm.hpp>

void run_DQGMRES(const std::string& matrix_path, double tol, int max_iter,
		int restart_iter, int k) {

    // If no restart, set restart_iter = max_iter
    if (restart_iter <= 0) {
	restart_iter = max_iter;
    } 

    // Set up nickname
    typedef composyx::DenseMatrix<double, 1> Vector;


    // Load matrix from mtx file
    composyx::SparseMatrixCSR<double, int> A = utils::load_matrix<composyx::SparseMatrixCSR<double, int>>(matrix_path);

    // Initialize all data

	// Preconditioenr
    Precond_Jacobi<double, Vector, composyx::SparseMatrixCSR<double, int>> M(A);
    //Precond_Identity<double, Vector, composyx::SparseMatrixCSR<double, int>> M(A);


    	// System vectors x_0, b
/*
    int n = n_rows(A);
    Vector x(n), b(n);
    for (int k = 0; k < n; ++k) {
        x(k) = 0.;
        b(k) = 1.0;
    }
    b = A * b;    
*/
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    int n = n_rows(A);
    Vector x(n), b(n), y(n);
    for (int k = 0; k < n; ++k) {
        y(k) = dist(gen);
	x(k) = 0;
    }
    b = A * y;
	
    double normb = norm(b);
    double normA = approximate_mat_norm(A);

	// Basis V, P and H
    composyx::DenseMatrix<double> V(n, restart_iter + 1); 
    composyx::DenseMatrix<double> P(n, restart_iter + 1);
    composyx::DenseMatrix<double> H(restart_iter + 1, restart_iter);

    // Run sGMRES
    auto start = std::chrono::high_resolution_clock::now();
    int i = DQGMRES(A, normA, x, b, normb, M, V, H, P, max_iter, restart_iter, tol, k);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    

    // Display results
    if (i == 0) {
	std::cout << "Converged in " << max_iter << " iterations\n";
    } else {
        std::cout << "Did not converge\n";
    }

    double backward_error = norm(b - A * x) / (normA * norm(x) + normb);
    std::cout << "||b - A * x|| / (||A||||x|| + ||b||): " << backward_error << std::endl;   
    std::cout << "Time for operation: " << elapsed.count() << " seconds\n";
}



void run_sGMRES(const std::string& matrix_path, double tol, int max_iter,
		int restart_iter, int k) {

    // If no restart, set restart_iter = max_iter
    if (restart_iter <= 0) {
	restart_iter = max_iter;
    } 

    // Set up nickname
    typedef composyx::DenseMatrix<double, 1> Vector;


    // Load matrix from mtx file
    composyx::SparseMatrixCSR<double, int> A = utils::load_matrix<composyx::SparseMatrixCSR<double, int>>(matrix_path);

    // Initialize all data

	// Preconditioenr
    Precond_Jacobi<double, Vector, composyx::SparseMatrixCSR<double, int>> M(A);
    //Precond_Identity<double, Vector, composyx::SparseMatrixCSR<double, int>> M(A);


    	// System vectors x_0, b
/*
    int n = n_rows(A);
    Vector x(n), b(n);
    for (int k = 0; k < n; ++k) {
        x(k) = 0.;
        b(k) = 1.0;
    }
    b = A * b;    
*/
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    int n = n_rows(A);
    Vector x(n), b(n), y(n);
    for (int k = 0; k < n; ++k) {
        y(k) = dist(gen);
	x(k) = 0;
    }
    b = A * y;
	
    double normb = norm(b);
    double normA = approximate_mat_norm(A);

	// Basis V
    composyx::DenseMatrix<double> V(n, restart_iter + 1); 

	// Sketch S, and QR factorization Q, R
    int sketch_size = 2 * (restart_iter + 1);
/*
    composyx::DenseMatrix<double> S(sketch_size, n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < sketch_size; ++i) {
    	for (int j = 0; j < n; ++j) {
            S(i, j) = dist(gen);
    	}
    }
*/
    SRHT<Vector> S(sketch_size, n);

    composyx::DenseMatrix<double> Q(sketch_size, restart_iter + 1);
    composyx::DenseMatrix<double> R(restart_iter + 1, restart_iter + 1);
    
    // Run sGMRES
    auto start = std::chrono::high_resolution_clock::now();
    int i = sGMRES(A, normA, x, b, normb, M, V, S, Q, R, max_iter, restart_iter, tol, k);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    

    // Display results
    if (i == 0) {
	std::cout << "Converged in " << max_iter << " iterations\n";
    } else {
        std::cout << "Did not converge\n";
    }

    double backward_error = norm(b - A * x) / (normA * norm(x) + normb);
    std::cout << "||b - A * x|| / (||A||||x|| + ||b||): " << backward_error << std::endl;   
    std::cout << "Time for operation: " << elapsed.count() << " seconds\n";
}


void print_help() {
    std::cout << R"(
Usage: ./main [options]

Options:
  -m, --matrix <file>           	Path to the matrix (.mtx) file [default: ./data/1138_bus.mtx]
  -t, --tol <float>             	GMRES convergence tolerance [default: 1e-9]
  -i, --max_iter <int>          	Maximum number of GMRES iterations [default: 1500]
  -ri, --restart-iter <int>		Number of iterations between restarts, no restart if 0 [default: 0]
  -k, 					k-truncated Arnoldi process parameter [default: 4]
  -h, --help                		Show this help message and exit
)";
}


int main(int argc, char* argv[]) {

    // Default parameter
    std::string matrix_path = "./data/small/1138_bus.mtx";
    double tol = 1e-9;
    int max_iter = 1500;
    int restart_iter = 0;
    int k = 4;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
	if ((arg == "--matrix" || arg == "-m") && i + 1 < argc)
            matrix_path = std::string(argv[++i]);
	else if ((arg == "--tol" || arg == "-t") && i + 1 < argc)
            tol = std::stod(argv[++i]);
	else if ((arg == "--max_iter" || arg == "-i") && i + 1 < argc)
            max_iter = std::stoi(argv[++i]);
	else if ((arg == "--restart_iter" || arg == "-ri") && i + 1 < argc)
            restart_iter = std::stoi(argv[++i]);
	else if ((arg == "-k") && i + 1 < argc)
            k = std::stoi(argv[++i]);
	else if (arg == "--help" || arg == "-h") {
            print_help();
	    return 0;
	} else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    //run_sGMRES(matrix_path, tol, max_iter, restart_iter, k);
    run_DQGMRES(matrix_path, tol, max_iter, restart_iter, k);

    return 0;
}
