# Auto-tuned GEMM kernels for Intel Skylake Processors

Auto-tuned Double-precision General Matrix-Matrix Multiplication kernels for Intel® Xeon® Gold 6140 Processor.

# Build

To build an executable, run `make skl`.  
The executable can be used to benchmark our kernel.

# Run Benchmark

To benchmark our kernel, run the executable generated in `out` directory.  
```
Usage: ./out/dgemm_flops_skl_[name].out NumThreads Layout(Row/Col) TransA(T/N) TransB(T/N) M N K alpha beta iter
```
For example, `./out/dgemm_flops_skl_kernel.mc.out 36 Col N N 20000 20000 20000 -1.0 1.0 10`.

You can also benchmark the shared library using [benchmark-dgemm](https://github.com/lshqqytiger/benchmark-dgemm).  
```
$ benchmark kernel_skl/kernel.oc.c -m 2000 -n 2000 -k 2000 --layout COL --alpha -1.0 --compiler-args "-O3 -march=native -lnuma -Iinclude -fopenmp -qmkl -qopenmp -wd3950 -fPIC" --override-compiler-args
```

# Performance

## Single-core

TBA

## Multi-core

TBA
