#!/usr/bin/env bash

clang parallel/csrc/test.cu -fplugin=/opt/enzyme/lib/ClangEnzyme-18.so -O2 --cuda-gpu-arch=sm_70 -lcudart -L/usr/local/cuda-11.8/lib64 -o parallel/csrc/a.exe &&
parallel/csrc/a.exe
echo "done."