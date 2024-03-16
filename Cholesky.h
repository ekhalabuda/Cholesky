#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>
typedef double FP;
class Cholesky{
private:
    class matrix{
    public:
        FP* m;
        size_t matrix_size;
        matrix() = default;
        matrix(size_t N_size):matrix_size(N_size){
            m = new FP[N_size*(N_size+1)/2];
        };
        ~matrix(){
            delete[] m;
        };
        inline FP& operator()(int row, int col){
            return m[row+col*(col+1)/2];
        };
        void Print(){
            std::cout << "Lower / symmetric matrix:" << std::endl;
            for (size_t i = 0; i < matrix_size; i++) {
                for(size_t j = 0; j <= i; j++)
                    std::cout << m[j+i*(i+1)/2] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };
    size_t N;
    size_t d_size;
    size_t c_size;
    FP norm_max = 0;
    FP aij_with_norm_max;
    matrix A;
    matrix A_check;
    matrix LA;
    void multiplication_matrix(FP* B, matrix& L, FP* dest);
    void multiplication_and_subtraction_matrix(FP* A, FP* B, matrix& dest);
    void multiplication_lower(FP* B, matrix& M, FP* dest);
    void inverse_matrix(matrix& M);
    void inverse_lower(matrix& L);
    void Cholesky_dec_block();
    void Cholesky_dec(matrix& M, matrix& L);
    void Positive_definite_symmetric_matrix_generator();
    void Decomposition_check();
    void Print_symmetric_matrix(matrix& M);
    void Print_lower(matrix& L);
    void Print_check_matrix();
    void print(FP* B, size_t n);
    void Print_matrix(FP* B);
    void Print_trans_matrix(FP* B);

public:
    Cholesky(size_t N_size = 5): N(N_size), d_size(N_size / 2), 
    c_size(N_size - N_size / 2), A(N_size), A_check(N_size), LA(N_size){}
    ~Cholesky(){}
    void analyse();

};