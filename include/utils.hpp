#pragma once

#include <cstddef>
#include <string>
#include <composyx.hpp>
#include <cmath>

#include <mkl_cblas.h>

namespace utils {

template<class MatrixCSR>
MatrixCSR load_matrix(const std::string file_name) {
    MatrixCSR A;
    A.from_matrix_market_file(file_name);

    return A;
}

template <class Real> Real sqrt(const Real &v) { return Real(std::sqrt(v)); }

template <class Vector> typename Vector::value_type norm(const Vector &v) {
    return sqrt(cblas_ddot(v.size(), v.data(), 1, v.data(), 1));
}


template<class MatrixCSR, class Vector>
void original_spmv(const MatrixCSR& mat, const double* v, Vector& r) {
	std::size_t n = n_rows(mat);
	const int* i_ptr = mat.get_i_ptr();
	const int* j_ptr = mat.get_j_ptr();
	const double* v_ptr = mat.get_v_ptr();

	#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0;

            for (int k = i_ptr[i]; k < i_ptr[i + 1]; ++k) {
               	sum += v_ptr[k] * v[j_ptr[k]];
            }

            r(i) = sum;
        }
    }

template <class Matrix, class Vector>
void Update(Vector& x, int k, Matrix& h, const Vector& s, const Matrix &V) {
    Vector y(s);

    // Backsolve: 
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, k + 1, 1, 1., h.data(), n_rows(h), y.data(), k + 1);
    
    cblas_dgemv(CblasColMajor, CblasNoTrans, n_rows(V), k + 1, 1., V.data(), n_rows(V), y.data(), 1, 1., x.data(), 1);
    
}

}
