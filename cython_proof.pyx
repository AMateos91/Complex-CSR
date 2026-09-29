# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

# Importación explícita de funciones matemáticas puras de C (libres de GIL)
from libc.math cimport sqrt

# Estructura limpia en C nativo
cdef struct CComplexBall:
    double mid_re
    double mid_im
    double radius
    double mag

# Añadimos 'nogil' al final de la firma para dar permiso de ejecución sin el hilo de Python
cdef inline CComplexBall c_mul(CComplexBall a, CComplexBall b) noexcept nogil:
    cdef CComplexBall res
    res.mid_re = (a.mid_re * b.mid_re) - (a.mid_im * b.mid_im)
    res.mid_im = (a.mid_re * b.mid_im) + (a.mid_im * b.mid_re)
    res.radius = (a.mag * b.radius) + (b.mag * a.radius) + (a.radius * b.radius)
    res.mag = sqrt(res.mid_re * res.mid_re + res.mid_im * res.mid_im)
    return res

# Añadimos 'nogil' aquí también
cdef inline CComplexBall c_add(CComplexBall a, CComplexBall b) noexcept nogil:
    cdef CComplexBall res
    res.mid_re = a.mid_re + b.mid_re
    res.mid_im = a.mid_im + b.mid_im
    res.radius = a.radius + b.radius
    res.mag = sqrt(res.mid_re * res.mid_re + res.mid_im * res.mid_im)
    return res

# Función principal expuesta a Python
def spmv_csr_cython(
    double[:] values_re, 
    double[:] values_im, 
    double[:] values_rad, 
    double[:] values_mag,
    long long[:] columns, 
    long long[:] row_pointers, 
    double[:] x_re, 
    double[:] x_im, 
    double[:] x_rad, 
    double[:] x_mag,
    long long dimension
):
    # Crear vectores de salida vacíos mediante NumPy
    cdef double[:] y_re = np.zeros(dimension, dtype=np.float64)
    cdef double[:] y_im = np.zeros(dimension, dtype=np.float64)
    cdef double[:] y_rad = np.zeros(dimension, dtype=np.float64)
    
    # Declaración explícita de variables locales al estilo C
    cdef long long i, k, col_idx
    cdef long long start_idx, end_idx
    cdef CComplexBall row_sum, mat_val, x_val, prod
    
    # Bucle crítico ejecutado a velocidad de C pura sin el bloqueo de hilos (GIL)
    with nogil:
        for i in range(dimension):
            row_sum.mid_re = 0.0
            row_sum.mid_im = 0.0
            row_sum.radius = 0.0
            row_sum.mag = 0.0
            
            start_idx = row_pointers[i]
            end_idx = row_pointers[i + 1]
            
            for k in range(start_idx, end_idx):
                col_idx = columns[k]
                
                # Asignación directa en registros de memoria rápidos
                mat_val.mid_re = values_re[k]
                mat_val.mid_im = values_im[k]
                mat_val.radius = values_rad[k]
                mat_val.mag = values_mag[k]
                
                x_val.mid_re = x_re[col_idx]
                x_val.mid_im = x_im[col_idx]
                x_val.radius = x_rad[col_idx]
                x_val.mag = x_mag[col_idx]
                
                # Al estar declaradas como 'nogil', ahora Cython permite llamarlas aquí sin problemas
                prod = c_mul(mat_val, x_val)
                row_sum = c_add(row_sum, prod)
            
            y_re[i] = row_sum.mid_re
            y_im[i] = row_sum.mid_im
            y_rad[i] = row_sum.radius
            
    return np.asarray(y_re), np.asarray(y_im), np.asarray(y_rad)
