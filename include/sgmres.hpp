#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <stdexcept>
#include "utils.hpp"
#include "chrono_logger.hpp"

#include <mkl_cblas.h>

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

template <class Real>
void ApplyPlaneRotation(Real& dx, Real& dy, Real& cs, Real& sn) {
    Real temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}

template <class Matrix, class Vector>
void Update(Vector& x, int k, Matrix& h, const Vector& s, const Matrix &V) {
    Vector y(s);

    // Backsolve: 
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, k + 1, 1, 1., h.data(), n_rows(h), y.data(), k + 1);    

    cblas_dgemv(CblasColMajor, CblasNoTrans, n_rows(V), k + 1, 1., V.data(), n_rows(V), y.data(), 1, 1., x.data(), 1);
    
}


template <class Real>
Real abs(Real x) {
    return (x > 0 ? x : -x);
}


// ========== sGMRES ========== //

template<class Operator, class Vector, class Matrix, class Sketch>
int sGMRES(Operator& A, double normA, Vector& x, Vector& b, double normb, Matrix& H, Matrix& V, Vector* v, Sketch S,
	   int& max_iter, int& restart_iter, double& tol) {

    // Initialization
    double resid;
    int i, j = 1; // Iterators
    int n = n_rows(A);
    Vector w(n), r(n);

    original_spmv(A, x, r);
    r = b - r;
    double beta = norm(r);

    double backward_error = beta / (normA * norm(x) + normb);
    if (backward_error <= tol) {
	tol = beta;
	max_iter = 0;
	return 0;
    }

    resid = beta;




    return 1;
}






