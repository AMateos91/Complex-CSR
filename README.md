# Complex-CSR Sparse Matrix-Vector Multiplication Framework

An efficient, multi-platform replication suite for sparse matrix-vector multiplication (SpMV) under dynamic arbitrary-precision complex interval (ball) arithmetic.

This repository hosts the hardware-accelerated baseline implementation used as a verification and validation benchmark suite for publication in **ACM Transactions on Mathematical Software (TOMS)**.

## Repository Contents

*   `STDM_Protected.cpp`: Multi-threaded C++ baseline driver benchmark utilizing the Boost Multiprecision MPFR backend.
*   `cython_proof.pyx`: Hardware-insulated, C-structured Cython extension routine bypassing the Python GIL.
*   `setup.py`: Setuptools configuration script for compiling the Cython extension.
*   `CSR.py`: High-throughput NumPy orchestration benchmark driver providing synthetic data generation and performance logging.
*   `Makefile`: Multi-platform GNU automated construction build file.
*   `LICENSE`: Official MIT Open-Source software license.

## System Prerequisites

To build and execute this framework, your platform must provide:
*   **C++ Compiler:** GCC \(\ge\) 9.3 or Clang \(\ge\) 12.0 (with C++17 and OpenMP support).
*   **Libraries:** GNU MPFR (\(\ge\) 4.0) and GNU GMP (\(\ge\) 6.1).
*   **Python:** Version \(\ge\) 3.8 (Tested up to Python 3.13).
*   **Packages:** `numpy` (\(\ge\) 1.20) and `cython` (\(\ge\) 3.0).

## Compilation and Benchmarking Instructions

All operations are automated via the bundled multi-platform `Makefile`.

### 1. Build All Components
To compile both the native C++ binary and the optimized Cython shared objects concurrently:
```bash
make all
```

### 2. Run Replication Driver Suite
To execute the automated non-interactive benchmarks and dump replication metrics (ACM Reproducibility Protocol):
```bash
make run_all
```
This pipes test logs and execution outputs straight into `expected_output_cpp.txt` and `expected_output_cython.txt`.

### 3. Cleanup Workspace
To purge all compiled binary objects, temporary directory structures, and cached output files before re-packing:
```bash
make clean
```

## License
This project is licensed under the terms of the MIT License.
