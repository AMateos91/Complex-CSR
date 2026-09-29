import numpy as np
import time
from cython_proof import spmv_csr_cython

dimension = 1000000
nnz = 1250000 

values_re = np.random.uniform(0.5, 1.5, nnz)
values_im = np.random.uniform(0.5, 1.5, nnz)
values_rad = np.random.uniform(1e-35, 1e-32, nnz)
values_mag = np.sqrt(values_re**2 + values_im**2)

columns = np.random.randint(0, dimension, nnz, dtype=np.int64)
row_pointers = np.sort(np.random.randint(0, nnz, dimension + 1, dtype=np.int64))
row_pointers[0], row_pointers[-1] = 0, nnz

x_re = np.ones(dimension)
x_im = np.zeros(dimension)
x_rad = np.full(dimension, 1e-30)
x_mag = np.ones(dimension)

# Medición del rendimiento optimizado
t0 = time.perf_counter()
y_re, y_im, y_rad = spmv_csr_cython(
    values_re, values_im, values_rad, values_mag,
    columns, row_pointers,
    x_re, x_im, x_rad, x_mag,
    dimension
)
print(f" Done! Running time of Cython SpMV for {dimension}x{dimension} matrix: {time.perf_counter() - t0:.4f} seconds")
