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

template<class Operator, class Vector, class Matrix, class Preconditioner>
int DQGMRES(Operator& A, double normA, Vector& x, Vector& b, double normb, Preconditioner& M, Matrix& V, Matrix& H, Matrix& P,
	int& max_iter, int& restart_iter, double& tol, const int& k) {

    // Initialization
    double resid;
    int i, j = 1; // Iterators
    int n = n_rows(A);

    Vector w(n), r(n), tx(n), z(n), cs(restart_iter + 1), sn(restart_iter + 1), g(restart_iter + 1);

    //M.solve(x, tx);

    r = b - A * x;
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

	    // Apply Preconditioner
	    cblas_dcopy(n, V.data() + n * i, 1, z.data(), 1);
	    M.solve(z, z);

	    // SpMV
	    w = A * z;

	    // k-truncated Arnoldi
	    for (int iter = std::max(0, i - k); iter <= i; ++iter) {
		H(iter, i) = cblas_ddot(n, w.data(), 1, V.data() + n * iter, 1);
		cblas_daxpy(n, -H(iter, i), V.data() + n * iter, 1, w.data(), 1);
	    }
	    H(i + 1, i) = norm(w);
	    cblas_dcopy(n, w.data(), 1, V.data() + n * (i + 1), 1);
	    cblas_dscal(n, 1.0 / H(i + 1, i), V.data() + n * (i + 1), 1);

	    // Rotation
	    for (int iter = std::max(0, i - k); iter < i; ++iter) {
    		ApplyPlaneRotation(H(iter, i), H(iter+1, i), cs(iter), sn(iter));
	    }
	    GeneratePlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
	    ApplyPlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));

	    ApplyPlaneRotation(g(i), g(i + 1), cs(i), sn(i));

	    // Update x_k
	    cblas_dcopy(n, z.data(), 1, P.data() + n * i, 1); // P_m = V_m
	    for (int iter = std::max(0, i - k); iter < i; ++iter) {
		cblas_daxpy(n, -H(iter, i), P.data() + n * iter, 1, P.data() + n * i, 1);
	    }
	    cblas_dscal(n, 1.0 / H(i, i), P.data() + n * i, 1);
	
	    cblas_daxpy(n, g(i), P.data() + n * i, 1, x.data(), 1);

	   // Stopping criterion
	   resid = norm(b - A * x);
	   backward_error = resid /(normb + norm(x) * normA);
		
	   std::cout << j << " " << i << " " << backward_error << " (" << resid <<")" << std::endl;
	
	   if (backward_error < tol) {
		max_iter = j;
		tol = resid;
		return 0;
	   }
	 
	} // End for i
	
	r = b - A * x;
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
