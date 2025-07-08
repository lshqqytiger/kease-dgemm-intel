# Auto-tuned GEMM kernels for Intel KNL Processors

Auto-tuned Double-precision General Matrix-Matrix Multiplication kernels for Intel® Xeon Phi™ Processor 7250.

# Build

To build an executable, run `make knl`.  
The executable can be used to benchmark our kernel.

# Run Benchmark

To benchmark our kernel, run the executable generated in `out` directory.  
```
Usage: ./out/dgemm_flops_knl_[name].out NumThreads Layout(Row/Col) TransA(T/N) TransB(T/N) M N K alpha beta iter
```
For example, `./out/dgemm_flops_knl_kernel.mc.out 68 Col N N 6000 6000 6000 -1.0 1.0 10`.

You can also benchmark the shared library using [benchmark-dgemm](https://github.com/lshqqytiger/benchmark-dgemm).  
```
$ benchmark kernel_knl/kernel.oc.c -m 600 -n 600 -k 600 --layout COL --alpha -1.0 --compiler-args "-O3 -march=native -lnuma -Iinclude -fopenmp -qmkl -qopenmp -wd3950 -fPIC" --override-compiler-args --numa-node 1
```

# Performance

## Single-core

TBA

## Multi-core

TBA
