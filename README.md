# GMRES-Variants
A few GMRES variants

This repository provides a prototype of different variants of GMRES in C++ using the Composyx library. In particular, it offers:
- Sketched GMRES (sGMRES)
- Direct Quasi-GMRES (DQGMRES)

## Content
Currently, are available:
- Preconditioner
  - Identity
  - Jacobi
- Sketching matrix (for sGMRES)
  - Subsampled Random Hadamar Transform (SRHT)
 
## How to run
Compile using the script:
```
./build-pure-main.sh
```
It requires guix (for Composyx).

It can then be run with:
```
./bin/main
```

The argument `-h` will display available arguments for the program:
```
./bin/main -h
```

The program loads matrices from `mtx` files and expect them to be sparse (You can pick them up from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

In `src/main.cpp` one might want to change the function called by the `main` function in order to use the desired GMRES variant.
