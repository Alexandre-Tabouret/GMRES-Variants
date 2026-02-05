#pragma once

#include <random>

#include <composyx.hpp>

template<class Vector>
class SketchingMatrix {
public:
    virtual ~SketchingMatrix() = default;

    SketchingMatrix(const std::size_t s, const std::size_t n): _s(s), _n(n) {}

    virtual void sketch(const double* v, Vector& res) = 0;

    friend std::size_t n_rows(const SketchingMatrix& S) {
        return S._s;
    }

protected:
    const std::size_t _s;
    const std::size_t _n;

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
	#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i)
	    _E(i) = (rand() & 1) ? 1 : -1;

	_D = Vector(s);
	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<int> perm(n);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(), rng);
	#pragma omp parallel for
	for (std::size_t i = 0; i < s; ++i)
	    _D(i) = perm[i];

	_work = Vector(n);
    }
    
    void sketch(const double* v, Vector& res) {

	// Apply E
	#pragma omp parallel for
        for (std::size_t i = 0; i < this->_n; ++i)
	    _work(i) = _E(i) * v[i];

	// Apply H
	this->apply_hadamard();

	// Apply D and scaling
	double scale = std::sqrt(static_cast<double>(this->_n / this->_s));
	#pragma omp parallel for
	for (std::size_t i = 0; i < this->_s; ++i)
            res(i) = scale * _work((std::size_t) _D(i));	

    }

private:
    Vector _E;
    Vector _D;

    Vector _work; // working vector for sketching allocated once

    void apply_hadamard() {
	size_t N = 1;
	while (N < this->_n) N <<= 1;	

	std::vector<double> tmp(N, 0.0);
	#pragma omp parallel for
    	for (size_t i = 0; i < this->_n; ++i) {
    	    tmp[i] = _work(i);
    	}

	for (size_t len = 1; len < N; len <<= 1) {
   	     for (size_t i = 0; i < N; i += (len << 1)) {
        	    for (size_t j = 0; j < len; ++j) {
                	double u = tmp[i + j];
                	double v = tmp[i + j + len];
                	tmp[i + j] = u + v;
                	tmp[i + j + len] = u - v;
            	    }
        	}
    	}

	#pragma omp parallel for
	for (size_t i = 0; i < this->_n; ++i)
             _work(i) = tmp[i];

    }
};
