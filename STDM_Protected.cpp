#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <limits>
#include <iomanip> // Necesario para std::setprecision
#include <string>  // Necesario para std::stoi
#include <omp.h>
#include <boost/multiprecision/mpfr.hpp> // Backend real para precisión dinámica

// Definición de flotante con precisión dinámica por asignación en el heap
typedef boost::multiprecision::mpfr_float DynamicFloat;

struct DynamicComplexBall {
    DynamicFloat mid_re;
    DynamicFloat mid_im;
    DynamicFloat radius;
    DynamicFloat mag;

    // Constructor por defecto utilizando inicializadores del tipo dinámico
    DynamicComplexBall() 
        : mid_re(DynamicFloat(0)), mid_im(DynamicFloat(0)), radius(DynamicFloat(0)), mag(DynamicFloat(0)) {}

    // Constructor parametrizado con conversiones estrictas
    DynamicComplexBall(DynamicFloat re, DynamicFloat im, DynamicFloat r) 
        : mid_re(re), mid_im(im), radius(r) {
        
        if (radius < DynamicFloat(0)) {
            radius = DynamicFloat(0);
        }
        // mpfr gestiona de forma nativa la precisión dinámica en funciones matemáticas externas
        mag = boost::multiprecision::sqrt(mid_re * mid_re + mid_im * mid_im);
    }

    DynamicComplexBall operator +(const DynamicComplexBall & b) const {
        DynamicFloat res_re = this->mid_re + b.mid_re;
        DynamicFloat res_im = this->mid_im + b.mid_im;
        
        // MPFR permite extraer el épsilon dinámico correspondiente a los bits asignados actualmente
        DynamicFloat eps = std::numeric_limits<DynamicFloat>::epsilon();
        DynamicFloat prop_err = (boost::multiprecision::abs(res_re) + boost::multiprecision::abs(res_im)) * eps;
        DynamicFloat res_radius = this->radius + b.radius + prop_err;
        
        return DynamicComplexBall(res_re, res_im, res_radius);
    }

    DynamicComplexBall operator *(const DynamicComplexBall & b) const {
        DynamicFloat new_re = (this->mid_re * b.mid_re) - (this->mid_im * b.mid_im);
        DynamicFloat new_im = (this->mid_re * b.mid_im) + (this->mid_im * b.mid_re);
        
        DynamicFloat eps = std::numeric_limits<DynamicFloat>::epsilon();
        DynamicFloat prop_err = (boost::multiprecision::abs(new_re) + boost::multiprecision::abs(new_im)) * eps;
        DynamicFloat new_radius = (this->mag * b.radius) + (b.mag * this->radius) + (this->radius * b.radius) + prop_err;
        
        return DynamicComplexBall(new_re, new_im, new_radius);
    }

    DynamicComplexBall & operator +=(const DynamicComplexBall & b) {
        *this = *this + b;
        return *this;
    }
};

void spmv_complex_csr_openmp(
    std::vector<DynamicComplexBall>& Y_base,
    const std::vector<DynamicComplexBall>& V_values,
    const std::vector<long>& C_columns,
    const std::vector<long>& P_pointers,
    const std::vector<DynamicComplexBall>& X_vector,
    long dimension)
{
    omp_set_num_threads(4);

    #pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < dimension; i++)
    {
        DynamicComplexBall row_sum(DynamicFloat(0), DynamicFloat(0), DynamicFloat(0));
        long start_idx = P_pointers[i];
        long end_idx = P_pointers[i + 1];

        for (long k = start_idx; k < end_idx; k++)
        {
            long col_idx = C_columns[k];
            row_sum += V_values[k] * X_vector[col_idx];
        }
        Y_base[i] = row_sum;
    }
}

int main(int argc, char* argv[])
{
    // =======================================================================
    // LECTURA DINÁMICA DE PRECISIÓN DESDE ARGV
    // =======================================================================
    unsigned int precision_digits = 50; // Valor por defecto seguro

    if (argc > 1) {
        try {
            precision_digits = std::stoi(argv[1]);
            if (precision_digits <= 0) {
                std::cerr << "[-] Error: La precision debe ser un entero positivo. Usando 50 por defecto.\n";
                precision_digits = 50;
            }
        } catch (const std::exception& e) {
            std::cerr << "[-] Error al procesar argumento numérico: " << e.what() << ". Usando 50 por defecto.\n";
            precision_digits = 50;
        }
    } else {
        std::cout << "[*] Nota: No se especifico precision. Sintaxis: ./programa <digitos>. Usando 50 por defecto.\n";
    }

    // Aplicar la precisión global de MPFR para las instancias dinámicas de este hilo
    DynamicFloat::default_precision(precision_digits);

    long dimension = 50000; 
    double fixed_density = 0.0005;

    std::cout << "=======================================================================\n";
    std::cout << " NATIVE C++ RUNTIME SCANNER : COMPLEX-CSR MPFR DYNAMIC ARITHMETIC\n";
    std::cout << "=======================================================================\n";
    std::cout << " Runtime Precision Set to : " << precision_digits << " decimal digits.\n";
    std::cout << " Generating synthetic environment for dimension : " << dimension << "x" << dimension << " ...\n";

    std::vector<DynamicComplexBall> Y_base(dimension);
    std::vector<DynamicComplexBall> X_vector(dimension);

    for (long i = 0; i < dimension; i++) {
        X_vector[i] = DynamicComplexBall(DynamicFloat(1), DynamicFloat(0), DynamicFloat("1e-30"));
    }

    std::vector<DynamicComplexBall> V_values;
    std::vector<long> C_columns;
    std::vector<long> P_pointers;
    P_pointers.push_back(0);

    std::mt19937 generator(1337);
    std::uniform_real_distribution<double> coin_flip(0.0, 1.0);
    std::uniform_real_distribution<double> val_midpoint(0.5, 1.5);
    std::uniform_real_distribution<double> error_radius(1e-35, 1e-32);

    long active_elements = 0;
    for (long i = 0; i < dimension; i++)
    {
        for (long j = 0; j < dimension; j++)
        {
            if (coin_flip(generator) < fixed_density)
            {
                DynamicFloat real_mid = DynamicFloat(val_midpoint(generator));
                DynamicFloat imag_mid = DynamicFloat(val_midpoint(generator));
                DynamicFloat r_bound = DynamicFloat(error_radius(generator));

                DynamicComplexBall val(real_mid, imag_mid, r_bound);
                V_values.push_back(val);
                C_columns.push_back(j);
                active_elements++;
            }
        }
        P_pointers.push_back(active_elements);
    }

    double real_density_calculated = ((double)active_elements / ((double)dimension * dimension)) * 100.0;

    std::cout << "CSR structure packaged successfully.\n";
    std::cout << " -> Active Elements (NNZ) : " << active_elements << "\n";
    std::cout << " -> Calculated Real Density : " << real_density_calculated << " %\n\n";
    std::cout << " Executing parallel SpMV multiplication (4 Threads - OpenMP) ..." << std::endl;

    double start_time = omp_get_wtime();
    spmv_complex_csr_openmp(Y_base, V_values, C_columns, P_pointers, X_vector, dimension);
    double end_time = omp_get_wtime();

    double total_execution_time = end_time - start_time;

    std::cout << "-----------------------------------------------------------------------\n";
    std::cout << " HARDWARE TELEMETRY RESULTS \n";
    std::cout << "-----------------------------------------------------------------------\n";
    std::cout << " Complex-CSR Parallel Time : " << total_execution_time << " seconds.\n";
    std::cout << " Algorithm Throughput : " << (double)dimension / total_execution_time << " rows/second.\n";
    
    std::cout << " Sample Error Radius in Y: " 
              << std::setprecision(precision_digits) << Y_base[0].radius 
              << " (Bit-consistent propagation)\n";
    std::cout << "=======================================================================\n";

    return 0;
}
