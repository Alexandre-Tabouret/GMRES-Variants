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
void original_spmv(const MatrixCSR& mat, const Vector& v, Vector& r) {
	std::size_t n = n_rows(mat);
	const int* i_ptr = mat.get_i_ptr();
	const int* j_ptr = mat.get_j_ptr();
	const double* v_ptr = mat.get_v_ptr();

	#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0;

            for (int k = i_ptr[i]; k < i_ptr[i + 1]; ++k) {
               	sum += v_ptr[k] * v(j_ptr[k]);
            }

            r(i) = sum;
        }
    }


}
