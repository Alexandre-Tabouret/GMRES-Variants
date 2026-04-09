#pragma once

#include <random>
#include <mkl_dfti.h>
#include <composyx.hpp>

template<class Vector>
class SketchingMatrix {
public:
    virtual ~SketchingMatrix() = default;

    SketchingMatrix(const std::size_t s, const std::size_t n): _s(s), _n(n) {}

    virtual void sketch(const double* v, Vector& res) = 0;

    virtual void sketch(const double* v, double* res) = 0;

    friend std::size_t n_rows(const SketchingMatrix& S) {
        return S._s;
    }

protected:
    const std::size_t _s;
    const std::size_t _n;

};

// Gaussian Sketching matrix S
// Size s * n
// S ~ N(0, 1/s)
template<class Vector, class Matrix>
class Gaussian: public SketchingMatrix<Vector> {
public:
    Gaussian(const std::size_t s, const std::size_t n): SketchingMatrix<Vector>(s, n) {
	_S = Matrix(s, n);

	// Fill S
	std::random_device rd;
        std::mt19937 gen(rd());
	//std::mt19937 gen(42);
	std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(s));
	for (size_t i = 0; i < s; ++i)
	    for (size_t j = 0; j < n; ++j)
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
    SRHT(const std::size_t s, const std::size_t n): SketchingMatrix<Vector>(s, n) {
 	_E = Vector(n);
	{
	    std::random_device rd;
	    std::mt19937 gen(rd());
    	    //std::mt19937 gen(42);
	    std::uniform_int_distribution<int> dist(0, 1);
	
            for (std::size_t i = 0; i < n; ++i)
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
	for (std::size_t i = 0; i < s; ++i)
	    _D(i) = perm[i];
	}
	
	size_t N = 1;
	while (N < n) N <<= 1;
	_work = Vector(N);
	_N = N;

	_scale = std::sqrt(static_cast<double>(this->_n) / (static_cast<double>(this->_s) * static_cast<double>(_N)));
    }
 
   void sketch(const double* v, Vector& res) {

	// Apply E
	#pragma omp parallel for
        for (std::size_t i = 0; i < this->_n; ++i)
	    _work(i) = _E(i) * v[i];

	// Apply H
	this->apply_hadamard();

	// Apply D and scaling
	#pragma omp parallel for
	for (std::size_t i = 0; i < this->_s; ++i)
            res(i) = _scale * _work((std::size_t) _D(i));	

    }

    void sketch(const double* v, double* res) {

	// Apply E
	#pragma omp parallel for
        for (std::size_t i = 0; i < this->_n; ++i)
	    _work(i) = _E(i) * v[i];
	
	// Fill the rest of the work vector with 0
	#pragma omp parallel for
	for (size_t i = this->_n; i < _N; ++i)
    	     _work(i) = 0.0;

	// Apply H
	this->apply_hadamard();

	// Apply D and scaling
	#pragma omp parallel for
	for (std::size_t i = 0; i < this->_s; ++i)
            res[i] = _scale * _work((std::size_t) _D(i));	

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
class SDCT: public SketchingMatrix<Vector> {
public:
    SDCT(std::size_t s, std::size_t n): SketchingMatrix<Vector>(s, n) {
      	_E = Vector(n);
	
	#pragma omp parallel
	{
	    //std::random_device rd;
	    //std::mt19937 gen(rd());
	    //int tid = omp_get_thread_num();
    	    std::mt19937 gen(42);
	    std::uniform_int_distribution<int> dist(0, 1);
	
	    #pragma omp for
            for (std::size_t i = 0; i < n; ++i)
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
	for (std::size_t i = 0; i < s; ++i)
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
        for (std::size_t i = 0; i < this->_n; ++i)
            _work(i) = _E(i) * v[i];

        // Step 2: Apply DCT-II
        //apply_dct();
        DftiComputeForward(_dct_handle, _work.data());

        // Step 3: Subsample and scale
        double scale = std::sqrt(static_cast<double>(this->_n / this->_s));
        #pragma omp parallel for
        for (std::size_t i = 0; i < this->_s; ++i)
            res(i) = scale * _work((std::size_t)_D(i));
    }

    void sketch(const double* v, double* res) {
	#pragma omp parallel for
        for (std::size_t i = 0; i < this->_n; ++i)
            _work(i) = _E(i) * v[i];

	//apply_dct();
	DftiComputeForward(_dct_handle, _work.data());

	double scale = std::sqrt(static_cast<double>(this->_n / this->_s));
        #pragma omp parallel for
        for (std::size_t i = 0; i < this->_s; ++i) {
            res[i] = scale * _work((std::size_t)_D(i));
	}
    }


private:
    Vector _E, _D, _work;
    DFTI_DESCRIPTOR_HANDLE _dct_handle;

    void apply_dct() {
        std::size_t n = this->_n;
        Vector tmp(n);

        #pragma omp parallel for
        for (std::size_t k = 0; k < n; ++k) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sum += _work(j) * cos(M_PI * (2*j + 1) * k / (2.0 * n));
            }
            tmp(k) = (k == 0 ? std::sqrt(1.0/n) : std::sqrt(2.0/n)) * sum;
        }

        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i)
            _work(i) = tmp(i);
    }
};
