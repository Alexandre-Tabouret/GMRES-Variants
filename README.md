# sGMRES
Sketched GMRES (sGMRES) prototype

This repository provides a prototype of sGMRES in C++ using the Composyx library.

## Content
Currently, are available:
- Preconditioner
  - Jacobi
- Sketching matrix
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
