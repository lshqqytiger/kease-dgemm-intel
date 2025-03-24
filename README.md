# Auto-tuned GEMM kernels for Intel KNL Processors

Auto-tuned Double-precision General Matrix-Matrix Multiplication kernels for Intel® Xeon Phi™ Processor 7250.

# Requirements

- [GNU Make](https://www.gnu.org/software/make)
- [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
- [OpenMP](https://www.openmp.org)
- [NUMA](https://github.com/numactl/numactl)
- [memkind](https://github.com/memkind/memkind)

# Build

To build an executable, run `make kernel.01.mc`.  
The executable can be used to benchmark our kernel.

# Run Benchmark

To benchmark our kernel on a single core, run `run_oc.sh` on the shell.  
`run_oc.sh` runs benchmark using following parameters and repeats 5 times for each M, N, K:
```
Number of threads: 1
Layout: Column-major
TransA: No
TransB: No
M, N, K: 100 to 1000 by increasing 100
Alpha: -1.0
Beta: 1.0
```

To benchmark our kernel on multiple cores, run `run_mc.sh` on the shell.  
`run_mc.sh` runs benchmark using following parameters and repeats 5 times for each M, N, K:
```
Number of threads: 68
Layout: Column-major
TransA: No
TransB: No
M, N, K: 1000 to 10000 by increasing 1000
Alpha: -1.0
Beta: 1.0
```

Instead, you can run benchmark by directly running the executable.  
```
Usage: ./out/[name].out NumThreads Layout(Row/Col) TransA(T/N) TransB(T/N) M N K alpha beta iter
```
For example, `./out/dgemm_flops_kernel.01.mc.out 68 Col N N 6000 6000 6000 -1.0 1.0 10`.

You can also benchmark the shared library using [benchmark-dgemm](https://github.com/lshqqytiger/benchmark-dgemm).

# Performance

## Single-core

TBA

## Multi-core

TBA
