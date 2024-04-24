#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_blacs.h>
typedef double FP;
// #define matrix_macr(matrix, row, col) ((matrix))[(row) + (col) * ((col) + 1) / 2]
class Cholesky {
private:
    class matrix {
    public:
        FP* m;
        const size_t matrix_size;
        matrix() = default;
        matrix(size_t N_size) :matrix_size(N_size) {
            m = static_cast<FP*>(malloc(N_size * (N_size + 1) / 2 * sizeof(FP)));
            std::memset(m, 0, N_size * (N_size + 1) / 2 * sizeof(FP));

        };
        ~matrix() {
            free(m);
        };
        inline FP& operator()(int row, int col) {
            return m[row + col * (col + 1) / 2];
        };
        inline const FP& operator()(int row, int col)const {
            return m[row + col * (col + 1) / 2];
        };
        void Print() {
            std::cout << "Lower / symmetric matrix:" << std::endl;
            for (size_t i = 0; i < matrix_size; i++) {
                for (size_t j = 0; j <= i; j++)
                    std::cout << m[j + i * (i + 1) / 2] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };
    const size_t N;
    const size_t d_size;
    size_t c_size;
    int64_t mkl_time;
    FP norm_max = 0;
    FP aij_with_norm_max;
    matrix A;
    matrix A_check;
    matrix LA;
    matrix D;
    FP* inverse;
    FP* B_copy;
    size_t* row_number_block;
    size_t* col_number_block;
    void double_multiplication_and_subtraction_matrix(size_t begin);
    void multiplication_and_substraction_matrix_DB(size_t begin);
    void multiplication_inverse_matrix(size_t begin);
    void multiplication_inverse_lower(size_t begin);
    void inverse_matrix(size_t begin);
    void inverse_lower(size_t begin);
    void Cholesky_dec_block( size_t n);
    void Cholesky_dec(size_t begin, size_t end);
    void Positive_definite_symmetric_matrix_generator();
    void Decomposition_check();
    void Print_symmetric_matrix(const matrix& M);
    void Print_lower(const matrix& L);
    void Print_check_matrix();
    void print(const FP* B);
    void Print_matrix(const FP* B);
    void Print_trans_matrix(const FP* B);
    void test_mkl();

public:
    Cholesky(size_t N_size, size_t len) : N(N_size), d_size(len),
        c_size(N_size - d_size), A(N_size), A_check(N_size), LA(N_size), D(d_size){
            inverse = static_cast<FP*>(malloc(d_size * d_size * sizeof(FP)));
            std::memset(inverse, 0.0, d_size * d_size * sizeof(FP));
            B_copy = static_cast<FP*>(malloc(d_size * c_size * sizeof(FP)));
            std::memset(B_copy, 0.0, d_size * c_size * sizeof(FP));
            int tmp = (c_size-1)/d_size+1;
            row_number_block = static_cast<size_t*>(malloc(tmp*(tmp+1)/2* sizeof(size_t)));
            col_number_block = static_cast<size_t*>(malloc(tmp*(tmp+1)/2* sizeof(size_t)));
        }
    ~Cholesky() {
        free(inverse);
        free(B_copy);
        free(row_number_block);
        free(col_number_block);
    }
    void analyse(std::ofstream& outputFile);
    void analyse();

};
