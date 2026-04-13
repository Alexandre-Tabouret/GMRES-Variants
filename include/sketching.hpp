#pragma once

#include <random>
#include <mkl_dfti.h>
#include <composyx.hpp>


template<class Vector>
class SketchingMatrix {
public:
    virtual ~SketchingMatrix() = default;

    SketchingMatrix(const uint32_t s, const uint32_t n): _s(s), _n(n) {}

    virtual void sketch(const double* v, Vector& res) = 0;

    virtual void sketch(const double* v, double* res) = 0;

    friend uint32_t n_rows(const SketchingMatrix& S) {
        return S._s;
    }

protected:
    const uint32_t _s;
    const uint32_t _n;

};

// Gaussian Sketching matrix S
// Size s * n
// S ~ N(0, 1/s)
template<class Vector, class Matrix>
class Gaussian: public SketchingMatrix<Vector> {
public:
    Gaussian(const uint32_t s, const uint32_t n): SketchingMatrix<Vector>(s, n) {
	_S = Matrix(s, n);

	// Fill S
	std::random_device rd;
        std::mt19937 gen(rd());
	//std::mt19937 gen(42);
	std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(s));
	for (uint32_t i = 0; i < s; ++i)
	    for (uint32_t j = 0; j < n; ++j)
           	_S(i,j) = dist(gen);
    }

    void sketch(const double* v, Vector& res) {
	cblas_dgemv(CblasColMajor, CblasNoTrans, this->_s, this->_n, 1.0, _S.data(), this->_s, v, 1, 0.0, res.data(), 1);	
    }

    void sketch(const double* v, double* res) {
	cblas_dgemv(CblasColMajor, CblasNoTrans, this->_s, this->_n, 1.0, _S.data(), this->_s, v, 1, 0.0, res, 1);
    }

private:
    Matrix _S;

};



// Susampled Random Hadamard Transform
// sqrt(n / s) D H E
// D: sumbsampling matrix
// H: Hadamard matrix
// E: Diagonal random sign matrix
template<class Vector>
class SRHT: public SketchingMatrix<Vector> {
public:
    SRHT(const uint32_t s, const uint32_t n): SketchingMatrix<Vector>(s, n) {
 	_E = Vector(n);
	{
	    std::random_device rd;
	    std::mt19937 gen(rd());
    	    //std::mt19937 gen(42);
	    std::uniform_int_distribution<int> dist(0, 1);
	
            for (uint32_t i = 0; i < n; ++i)
	    	_E(i) = dist(gen) ? 1 : -1;
	}
	{
	_D = Vector(s);
	std::random_device rd;
	std::mt19937 rng(rd());
	//std::mt19937 rng(42);
	std::vector<int> perm(n);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(), rng);
	
	#pragma omp parallel for
	for (uint32_t i = 0; i < s; ++i)
	    _D(i) = perm[i];
	}
	
	uint32_t N = 1;
	while (N < n) N <<= 1;
	_work = Vector(N);
	_N = N;

	_scale = std::sqrt(static_cast<double>(this->_n) / (static_cast<double>(this->_s) * static_cast<double>(_N)));
    }
 
   void sketch(const double* v, Vector& res) {

	// Apply E
	#pragma omp parallel for
        for (uint32_t i = 0; i < this->_n; ++i)
	    _work(i) = _E(i) * v[i];

	// Apply H
	this->apply_hadamard();

	// Apply D and scaling
	#pragma omp parallel for
	for (uint32_t i = 0; i < this->_s; ++i)
            res(i) = _scale * _work((uint32_t) _D(i));	

    }

    void sketch(const double* v, double* res) {

	// Apply E
	#pragma omp parallel for
        for (uint32_t i = 0; i < this->_n; ++i)
	    _work(i) = _E(i) * v[i];
	
	// Fill the rest of the work vector with 0
	#pragma omp parallel for
	for (size_t i = this->_n; i < _N; ++i)
    	     _work(i) = 0.0;

	// Apply H
	this->apply_hadamard();

	// Apply D and scaling
	#pragma omp parallel for
	for (uint32_t i = 0; i < this->_s; ++i)
            res[i] = _scale * _work((uint32_t) _D(i));	

    }


   
private:
    Vector _E;
    Vector _D;

    Vector _work; // working vector for sketching allocated once
    size_t _N; // Actual size of _work (power of 2)

    double _scale;

    void apply_hadamard() {
	for (size_t len = 1; len < _N; len <<= 1) {
	     #pragma omp parallel for schedule(static)
   	     for (size_t i = 0; i < _N; i += (len << 1)) {
        	    for (size_t j = 0; j < len; ++j) {
                	double u = _work[i + j];
                	double v = _work[i + j + len];
                	_work[i + j] = u + v;
                	_work[i + j + len] = u - v;
            	    }
        	}
    	}
    }
};

template<class Vector>
class SparseSign: public SketchingMatrix<Vector> {
public:
    SparseSign(uint32_t s, uint32_t n, unsigned int zeta): SketchingMatrix<Vector>(s, n) {

	std::vector<std::vector<std::pair<uint32_t, double>>> rows(s);

	// Generate matrix
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<uint32_t> dist_row(0, s - 1);
	std::uniform_int_distribution<int> dist_sign(0, 1);

	double scale = 1.0 / std::sqrt((double) zeta);

	for (uint32_t j = 0; j < n; ++j) {
	    for (uint32_t k = 0; k < zeta; ++k) {
	        uint32_t i = dist_row(gen);
	        double val = (dist_sign(gen) ? 1.0 : -1.0) * scale;
	        rows[i].emplace_back(j, val);
	    }
	}

	// Convert in CSR format
	_row_ptr.resize(s + 1);
        _row_ptr[0] = 0;

        for (uint32_t i = 0; i < s; ++i) {
            _row_ptr[i + 1] = _row_ptr[i] + rows[i].size();
        }

        const uint32_t nnz = _row_ptr[s];
        _col_ind.resize(nnz);
        _values.resize(nnz);

        uint32_t idx = 0;
        for (uint32_t i = 0; i < s; ++i) {
            for (const auto& entry : rows[i]) {
                _col_ind[idx] = entry.first;
                _values[idx]  = entry.second;
                ++idx;
            }
	}
    }

    void sketch(const double* v, Vector& res) {
	spmv(_row_ptr, _col_ind, _values, v, res.data());
    }

    void sketch(const double* v, double* res) {
	spmv(_row_ptr, _col_ind, _values, v, res);
    }


private:
    std::vector<uint32_t> _row_ptr;
    std::vector<uint32_t> _col_ind;
    std::vector<double> _values;

    void spmv(std::vector<uint32_t>& row_ptr, std::vector<uint32_t>& col_ptr, std::vector<double>& values, const double* v, double* res) {
       	const uint32_t* i_ptr = row_ptr.data();
        const uint32_t* j_ptr = col_ptr.data();
        const double* val_ptr = values.data();

        #pragma omp parallel for
        for (uint32_t i = 0; i < this->_s; ++i) {
            double sum = 0;

            for (uint32_t k = i_ptr[i]; k < i_ptr[i + 1]; ++k) {
                sum += val_ptr[k] * v[j_ptr[k]];
            }

            res[i] = sum;
        }
    }

};



template<class Vector>
class SDCT: public SketchingMatrix<Vector> {
public:
    SDCT(uint32_t s, uint32_t n): SketchingMatrix<Vector>(s, n) {
      	_E = Vector(n);
	
	#pragma omp parallel
	{
	    //std::random_device rd;
	    //std::mt19937 gen(rd());
	    //int tid = omp_get_thread_num();
    	    std::mt19937 gen(42);
	    std::uniform_int_distribution<int> dist(0, 1);
	
	    #pragma omp for
            for (uint32_t i = 0; i < n; ++i)
	    	_E(i) = dist(gen) ? 1 : -1;
	}

	_D = Vector(s);
	//std::random_device rd;
	//std::mt19937 rng(rd());
	std::mt19937 rng(42);
	std::vector<int> perm(n);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(), rng);
	
	#pragma omp parallel for
	for (uint32_t i = 0; i < s; ++i)
	    _D(i) = perm[i];

	_work = Vector(n);

	// Create MKL DCT-II plan
	_dct_handle = nullptr;
        DftiCreateDescriptor(&_dct_handle, DFTI_DOUBLE, DFTI_REAL, 1, n);
        DftiSetValue(_dct_handle, DFTI_FORWARD_SCALE, 1.0); // scale later
        DftiCommitDescriptor(_dct_handle);
    }

    
    ~SDCT() {
	if (_dct_handle != nullptr) {
    	    DftiFreeDescriptor(&_dct_handle);
            _dct_handle = nullptr;
	}
    }

    void sketch(const double* v, Vector& res) {
        // Step 1: Apply random signs
        #pragma omp parallel for
        for (uint32_t i = 0; i < this->_n; ++i)
            _work(i) = _E(i) * v[i];

        // Step 2: Apply DCT-II
        //apply_dct();
        DftiComputeForward(_dct_handle, _work.data());

        // Step 3: Subsample and scale
        double scale = std::sqrt(static_cast<double>(this->_n / this->_s));
        #pragma omp parallel for
        for (uint32_t i = 0; i < this->_s; ++i)
            res(i) = scale * _work((uint32_t)_D(i));
    }

    void sketch(const double* v, double* res) {
	#pragma omp parallel for
        for (uint32_t i = 0; i < this->_n; ++i)
            _work(i) = _E(i) * v[i];

	//apply_dct();
	DftiComputeForward(_dct_handle, _work.data());

	double scale = std::sqrt(static_cast<double>(this->_n / this->_s));
        #pragma omp parallel for
        for (uint32_t i = 0; i < this->_s; ++i) {
            res[i] = scale * _work((uint32_t)_D(i));
	}
    }


private:
    Vector _E, _D, _work;
    DFTI_DESCRIPTOR_HANDLE _dct_handle;

    void apply_dct() {
        uint32_t n = this->_n;
        Vector tmp(n);

        #pragma omp parallel for
        for (uint32_t k = 0; k < n; ++k) {
            double sum = 0.0;
            for (uint32_t j = 0; j < n; ++j) {
                sum += _work(j) * cos(M_PI * (2*j + 1) * k / (2.0 * n));
            }
            tmp(k) = (k == 0 ? std::sqrt(1.0/n) : std::sqrt(2.0/n)) * sum;
        }

        #pragma omp parallel for
        for (uint32_t i = 0; i < n; ++i)
            _work(i) = tmp(i);
    }
};
