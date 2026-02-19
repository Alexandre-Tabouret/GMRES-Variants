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

template<class Matrix, class Vector>
void back_substitution(const Matrix& R, const Vector& b, Vector& y, int n) {
    for (int i = n - 1; i >= 0; --i) {
        double sum = b(i);
        for (int j = i + 1; j < n; ++j)
            sum -= R(i, j) * y(j);
        y(i) = sum / R(i, i);
    }
}



template <class Real>
Real abs(Real x) {
    return (x > 0 ? x : -x);
}


// ========== sGMRES ========== //

template<class Operator, class Vector, class Matrix, class Sketch, class Preconditioner>
int sGMRES(Operator& A, double normA, Vector& x, Vector& b, double normb, Preconditioner& M, Matrix& V, 
	   Sketch& S, Matrix& QR,
	   int& max_iter, int& restart_iter, double& tol, const int& k) {

    // Initialization
    double resid, h;
    int i, j = 1; // Iterators
    int n = n_rows(A);
    int s = n_rows(S);
    Vector w(n), r(n), w_s(s), Sr_0(s), y(restart_iter), z(n), tx(n), QtSr_0(s), tau(restart_iter);
    M.solve(x, tx);

    original_spmv(A, tx.data(), r.data());
    r = b - r;
    double beta = norm(r);

    double backward_error = beta / (normA * norm(tx) + normb);
    
    if (backward_error <= tol) {
	tol = beta;
	max_iter = 0;
	return 0;
    }

    resid = beta;

    // Outer loop
    while (j <= max_iter) {
	// Initialize V
	cblas_dcopy(n, r.data(), 1, V.data(), 1);
        cblas_dscal(n, 1.0 / beta, V.data(), 1);
	
	// Initialize g = S * r_0
	S.sketch(r.data(), Sr_0);
	//double norm_Sr_0_squared = norm(Sr_0);
	//norm_Sr_0_squared *= norm_Sr_0_squared;

	// Inner loop
	for (i = 0; i < restart_iter && j <= max_iter; ++i, ++j) {

	    // Apply Preconditioner
	    M.solve(n, V.data() + n * i, z.data());

	    // SpMV
	    original_spmv(A, z.data(), w.data()); // w = A * z

	    // Sketch the new vector
	    S.sketch(w.data(), QR.data() + s * i);

	    // k-truncated Arnoldi (MGS)
	    for (int iter = std::max(0, i - k); iter <= i; ++iter) {
		h = cblas_ddot(n, w.data(), 1, V.data() + n * iter, 1);
		cblas_daxpy(n, -h, V.data() + n * iter, 1, w.data(), 1);
	    }
	    h = norm(w);
	    cblas_dcopy(n, w.data(), 1, V.data() + n * (i + 1), 1);
	    cblas_dscal(n, 1.0 / h, V.data() + n * (i + 1), 1);

	    // QR factorization update
	    if (i == 0) {
	        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, s, 1, QR.data(), s, tau.data());
	    } else {
		LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', s, 1, i, QR.data(), s, tau.data(), QR.data() + s * i, s);
		LAPACKE_dlarfg(s - i, QR.data() + s * i + i, QR.data() + s * i + i + 1, 1, tau.data() + i);
	    }
	
	    // Compute the estimated residual
	    cblas_dcopy(s, Sr_0.data(), 1, QtSr_0.data(), 1);
	    LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', s, 1, i+1, QR.data(), s, tau.data(), QtSr_0.data(), s);
            //double norm_QtSr_0_squared = cblas_ddot(i+1, QtSr_0.data(), 1, QtSr_0.data(), 1);
            //double resid_estimate = std::sqrt(norm_Sr_0_squared - norm_QtSr_0_squared);
            //double resid_estimate = std::sqrt(cblas_ddot(s - (i + 1), QtSr_0.data() + i + 1, 1, QtSr_0.data() + i + 1, 1));

	    // Update x for backward error
	    //if (i%10 == 0) {
            if (10) {
                tx = x;
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, i+1, 1, 1., QR.data(), s, QtSr_0.data(), i+1); // y = QtSr_0(0:i+1)
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, i+1, 1.0, V.data(), n, QtSr_0.data(), 1, 1.0, tx.data(), 1);
                M.solve(tx, tx);
		original_spmv(A, tx.data(), r.data());
                resid = norm(b - r);
            }

	    backward_error = resid / (normA * norm(tx) + normb); // eta_{A,b}

//std::cout << j << " " << i << " " << backward_error << " " << resid << std::endl;
	    if (backward_error < tol) {
		x = tx;
		max_iter = j;
		tol = resid;
		return 0;
	    }

	} // End for i

    	// Update before restart    	
	cblas_dcopy(s, Sr_0.data(), 1, QtSr_0.data(), 1);
	LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', s, 1, i, QR.data(), s, tau.data(), QtSr_0.data(), s);
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, i, 1, 1., QR.data(), s, QtSr_0.data(), i);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, i, 1.0, V.data(), n, QtSr_0.data(), 1, 1.0, x.data(), 1);
	

	M.solve(x, tx);
	original_spmv(A, tx.data(), r.data());
    	r = b - r;
    	beta = norm(r);
    	resid = beta;
	
	backward_error = resid / (normA * norm(tx) + normb); // eta_{A,b}

    	if (backward_error < tol) {
	    x = tx;
	    tol = resid;
            max_iter = j;
	    return 0;
     	}



    } // End while j

    // No convergence

    M.solve(x, x);
    tol = resid;
    return 1;
}


