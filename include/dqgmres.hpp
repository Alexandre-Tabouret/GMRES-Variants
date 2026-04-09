#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <stdexcept>
#include "utils.hpp"
#include "chrono_logger.hpp"

#include <mkl_cblas.h>
#include <mkl.h>

using namespace std;
using namespace utils;

// ========== Utilitary Functions ========== //

template <class Real>
void GeneratePlaneRotation(Real& dx, Real& dy, Real& cs, Real& sn) {
    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
    } else if (abs(dy) > abs(dx)) {
        Real temp = dx / dy;
        sn = 1.0 / sqrt(1.0 + temp * temp);
        cs = temp * sn;
    } else {
        Real temp = dy / dx;
        cs = 1.0 / sqrt(1.0 + temp * temp);
        sn = temp * cs;
    }
}

/*
template <class Real>
void GeneratePlaneRotation(Real& dx, Real& dy, Real& cs, Real& sn) {
    cs = dx / std::sqrt(dx * dx + dy * dy);
    sn = dy / std::sqrt(dx * dx + dy * dy);
}
*/

template <class Real>
void ApplyPlaneRotation(Real& dx, Real& dy, Real& cs, Real& sn) {
    Real temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}


// ========== DQGMRES ========== //

template<class Operator, class Vector, class Matrix, class Preconditioner, class Logger>
int DQGMRES(Operator& A, double normA, Vector& x, Vector& b, double normb, Preconditioner& M, Matrix& V, Matrix& H, Matrix& P,
	int& max_iter, int& restart_iter, double& tol, const int& k, Logger& logger) {

    Chrono_logger chrono_logger;

    // Initialization
    double resid;
    int i, j = 1; // Iterators
    int n = n_rows(A);

    Vector w(n), r(n), tx(n), z(n), cs(restart_iter + 1), sn(restart_iter + 1), g(restart_iter + 1);
    Vector H_delta(n); // For CGS2

    //M.solve(x, tx);
    original_spmv(A, x.data(), r.data());
    r = b - r;

    double beta = norm(r);
    
    double backward_error = beta / (normA * norm(x) + normb);

    if (backward_error <= tol) {
	tol = beta;
	max_iter = 0;
	return 0;
    }

    resid = beta;

    while (j <= max_iter) {
	// Initialize V
	cblas_dcopy(n, r.data(), 1, V.data(), 1);
        cblas_dscal(n, 1.0 / beta, V.data(), 1);
	g *= 0;
	g(0) = beta;

        // Inner loop
	for (i = 0; i < restart_iter && j <= max_iter; ++i, ++j) {

	    chrono_logger.checkpoint(); // 0
	    double sketching_time = chrono_logger.checkpoint(); // 1

	    // Apply Preconditioner z_i = M^-1 * v_i
	    M.solve(n, V.data() + n * i, P.data() + n * i); // Directly store the result in p_i as it will need it in its update

	    double precond_time = chrono_logger.checkpoint(); // 2

	    // SpMV
	    original_spmv(A, P.data() + n * i, w.data());

	    double spmv_time = chrono_logger.checkpoint(); // 3

	    // k-truncated Arnoldi
	    
	    for (int iter = std::max(0, i - k + 1); iter <= i; ++iter) {
		H(iter, i) = cblas_ddot(n, w.data(), 1, V.data() + n * iter, 1);
		cblas_daxpy(n, -H(iter, i), V.data() + n * iter, 1, w.data(), 1);
	    }
	    H(i + 1, i) = norm(w);
	    cblas_dcopy(n, w.data(), 1, V.data() + n * (i + 1), 1);
	    cblas_dscal(n, 1.0 / H(i + 1, i), V.data() + n * (i + 1), 1);
 
/*
	    int start = std::max(0, i - k + 1); // truncated range (last k vectors)
	    int ki = i - start + 1;             // number of vectors in the truncated window

	    cblas_dgemv(CblasColMajor, CblasTrans, n, ki, 1.0, V.data() + start * n, n, w.data(), 1, 0.0, H.get_vect_view(i).data() + start, 1);
	    cblas_dgemv(CblasColMajor, CblasNoTrans, n, ki, -1.0, V.data() + start * n, n, H.get_vect_view(i).data() + start, 1, 1.0, w.data(), 1);
	    
	    cblas_dgemv(CblasColMajor, CblasTrans, n, ki, 1.0, V.data() + start * n, n, w.data(), 1, 0.0, H_delta.data(), 1);
	    cblas_dgemv(CblasColMajor, CblasNoTrans, n, ki, -1.0, V.data() + start * n, n, H_delta.data(), 1, 1.0, w.data(), 1);

	    #pragma omp parallel for
	    for (int j = 0; j < ki; ++j)
	        H(start + j, i) += H_delta[j];
	    H(i + 1, i) = norm(w);
	    cblas_dcopy(n, w.data(), 1, V.data() + (i + 1) * n, 1);
	    cblas_dscal(n, 1.0 / H(i + 1, i), V.data() + (i + 1) * n, 1);
*/

	    double ortho_time = chrono_logger.checkpoint(); // 4

	    // Rotation
	    for (int iter = std::max(0, i - k); iter < i; ++iter) {
    		ApplyPlaneRotation(H(iter, i), H(iter+1, i), cs(iter), sn(iter));
	    }
	    GeneratePlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
	    ApplyPlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));

	    ApplyPlaneRotation(g(i), g(i + 1), cs(i), sn(i));

	    double facto_time = chrono_logger.checkpoint(); // 5

	    // Update x_k
	    //cblas_dcopy(n, z.data(), 1, P.data() + n * i, 1); // P_i = M^-1 V_i --> already stored in it.
	    for (int iter = std::max(0, i - k); iter < i; ++iter) {
		cblas_daxpy(n, -H(iter, i), P.data() + n * iter, 1, P.data() + n * i, 1);
	    }
	    cblas_dscal(n, 1.0 / H(i, i), P.data() + n * i, 1);
	
	    cblas_daxpy(n, g(i), P.data() + n * i, 1, x.data(), 1);
	
	    double update_time = chrono_logger.checkpoint(); // 6

	   // Stopping criterion
	   //original_spmv(A, x.data(), r.data());
	   //resid = norm(b - r);
	   //backward_error = resid /(normb + norm(x) * normA);
	
	   double resid_est = std::sqrt(std::max(i - k, 0) + 1) * std::abs(g(i+1));
	   backward_error = resid_est /(normb + norm(x) * normA);

	   double be_time = chrono_logger.checkpoint(); // 7
	   auto math_time = chrono_logger.get_duration(0, 7);
           chrono_logger.clear();
           logger.log_chrono(j, math_time, sketching_time, precond_time, spmv_time, ortho_time, facto_time, update_time, be_time);

	   if (backward_error < tol) {
		// Check if it truly converged
		original_spmv(A, x.data(), r.data());
           	resid = norm(b - r);
		backward_error = resid /(normb + norm(x) * normA);
		
		if (backward_error < tol) {
		    max_iter = j;
		    tol = resid;
		    return 0;
		}
	   }
	 
	} // End for i
	
	original_spmv(A, x.data(), r.data());
	r = b - r;
    	beta = norm(r);
    	resid = beta;
	
	backward_error = resid / (normA * norm(x) + normb);

	if (backward_error < tol) {
	    tol = resid;
            max_iter = j;
            return 0;
     	}

    } // End while j

    tol = resid;
    return 1; 

} // End function
