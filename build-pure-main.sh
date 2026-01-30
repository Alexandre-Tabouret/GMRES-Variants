#!/bin/sh

guix time-machine -C channels.scm -- shell --pure -D composyx-mkl composyx-mkl -- ./run_main.sh
