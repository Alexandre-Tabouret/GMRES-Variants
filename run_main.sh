#!/bin/sh

mkdir -p bin

g++ -g -Wall -Wextra -pedantic -std=c++23 -O3 -march=native\
  -o ./bin/main ./src/main.cpp \
  -I./include \
  -I$GUIX_ENVIRONMENT/include \
  -DCOMPOSYX_USE_LAPACKPP \
  -L$GUIX_ENVIRONMENT/lib \
  -lm -lquadmath -llapackpp -lblaspp -lmpi -fopenmp -lgomp \
  -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl \
  -Wmaybe-uninitialized \
