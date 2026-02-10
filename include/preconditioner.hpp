#pragma once

#include <stdexcept>


template <typename real, class Vector, class Matrix>
class Abstract_Precond {

public:
    virtual Vector solve(const Vector& v) const = 0;
    virtual void solve(const Vector& v, Vector& res) const = 0;
    virtual ~Abstract_Precond() = default;

};

// Identity Preconditioner (none)
template <typename real, class Vector, class Matrix>
class Precond_Identity : public Abstract_Precond<real, Vector, Matrix> {

public:
    Precond_Identity(const Matrix& /*A*/) {}

    Vector solve(const Vector& v) const override {
        Vector r = Vector(v.size());
        #pragma omp parallel for
        for (std::size_t i = 0; i < v.size(); i++) {
            r(i) = v(i);
        }
        return r;
    }
 
    void solve(const Vector& v, Vector& res) const override {
        #pragma omp parallel for
        for (std::size_t i = 0; i < v.size(); i++) {
            res(i) = v(i);
        }
    }

}; 


// Jacobi Preconditioner (Vector that represents a diagonal)
template <typename real, class Vector, class Matrix>
class Precond_Jacobi : public Abstract_Precond<real, Vector, Matrix> {

public:
    Precond_Jacobi(const Matrix& A) {
        if (n_rows(A) <= 0) {
            throw std::invalid_argument("Invalid matrix size");
        } else if (n_rows(A) == 0) {
            std::cout << "[JacobiPreconditioner(Matrix &A)] Warning: empty matrix" << std::endl;
        }
        p = Vector(n_rows(A));
	real a;
        for (std::size_t i = 0; i < n_rows(A); i++) {
	    try {
		a = A(i, i);
	    } catch (const std::invalid_argument& e) {
          	a = 0;
            }
            if (a == 0) {
                p(i) = 1;
            } else {
                p(i) = 1.0 / a;
            }
        }
        norm = p.norm();
    }

    // Returns this * vector
    Vector solve(const Vector& v) const override {
        Vector r = Vector(v.size());
        #pragma omp parallel for
	for (std::size_t i = 0; i < this->p.size(); i++) {
            r(i) = v(i) * p(i);
        }
        return r;
    }

    void solve(const Vector& v, Vector& res) const override {
        #pragma omp parallel for
        for (std::size_t i = 0; i < this->p.size(); i++) {
            res(i) = v(i) * p(i);
        }
    }


    const real& operator()(size_t i) const {
        if (i < this->p.size()) { // i being a size_t, it is necessary >= 0 so dont need to check
            return this->p(i);
        } else {
            throw std::out_of_range("index out of range");
        }
    }

private:
    Vector p;
    double norm;
};
