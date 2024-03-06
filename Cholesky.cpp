#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>
// Base algo
// N = 100 : 790 microseconds.
// Chebyshev's norm =  0.0000000000 in aij = 4679.7271500924
// The maximum error is 0.0000000000425618704875199223603516084370 % 
// 

const size_t n = 15; // Размерность матриц
class Cholesky{
private:

    class matrix{
    public:
        double* m;
        matrix() = default;
        matrix(size_t N_size){
            m = new double[N_size*(N_size+1)/2];
        };
        ~matrix(){
            delete[] m;
        };
        inline double& operator()(int row, int col){
            return row <= col ? m[row+col*(col+1)/2] : m[col+row*(row+1)/2];
        };
    };

    size_t N;
    matrix L;
    matrix A;
    matrix A_check;
    double norm_max = 0;
    double aij_with_norm_max;
    //for block method
    // D  B'
    // B  C
    // S = C - B'D(-1)B
    //   |
    //   |
    //   \/
    //    | L(D)        0    |
    //L = | BL(-1)(D)   L(s) |
    matrix D;
    //matrix C;
    //matrix S;
    double* B;
    size_t d_size;

public:
    Cholesky(size_t N_size = 5): N(N_size), L(N_size), A(N_size), A_check(N_size), D((N_size - (!(N_size%2)))/2){}
    ~Cholesky(){}
    void Cholesky_dec_block(){
        if(N%2==1){
            d_size = (N-1)/2;
        }else d_size = N/2;
        for(size_t i = 0; i < N; i++){
            std::memcpy(D.m + i*d_size, A.m + i*N, d_size*sizeof(double));
        }
    }
    void Cholesky_dec() {
        L(0,0) = sqrt(A(0,0));
        for(size_t j = 1; j < N; j++){
            size_t i = 0; 
            L(i,j) = A(i,j) / L(0,0);
        }
        for (size_t i = 1; i < N; i++){
            double sum = 0;
            for (size_t p = 0; p < i; p++){
                sum += (L(p,i) * L(p,i));
            }
            L(i,i) = sqrt(std::abs(A(i,i) - sum));
            for (size_t j = i + 1; j < N; j++){
                sum = 0;
                for (size_t p = 0; p < i; p++){
                    sum += L(p,i) * L(p,j);
                }
                L(i,j) = (A(i,j) - sum) / L(i,i);
            }
        }
    }

    void Positive_definite_symmetric_matrix_generator(){
        unsigned int seed = 123;
        std::mt19937_64 generator(seed);
        std::uniform_real_distribution<double> distribution(-100.0, 100.0);

        for(size_t i = 0; i < N; i++)
            for(size_t j = 0; j < i; j++){
                A(i,j) = distribution(generator);
            }

        for(size_t i = 0; i < N; i++){
            double s = 0; 
            for(size_t j = 0; j < N; j++){
                if(i != j){
                    s += std::abs(A(i,j));
                }
            }
            std::uniform_real_distribution<double> distribution_2(s, 1000.0);
            A(i,i) = distribution_2(generator);
        }
    }

    void Decomposition_check(){
        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j <= i; j++){
                double aij = 0;
                for(size_t k = 0; k <= j; k++){
                    aij += L(k,i) * L(k,j);
                }
                A_check(i,j) = aij;
                double tmp = std::abs(aij - A(i,j));
                if(tmp > norm_max){
                    norm_max = tmp;
                    aij_with_norm_max = aij;
                }
            }
        }
    }

    void Print_Matrix(){
        std::cout << "Positive Definite Symmetric Matrix (A):" << std::endl;
        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++){
                std::cout << A(i,j)<<' ';
            }
            std::cout<<std::endl;
        }
        std::cout << std::endl;
    }

    void Print_Lower(){
        std::cout << "Lower Triangular Matrix (L):" << std::endl;
        for (size_t i = 0; i < N; i++) {
            for(size_t j = 0; j <= i; j++)
                std::cout << L(i,j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void Print_Check_Matrix(){
        std::cout << "Check Matrix (A = LLт):" << std::endl;
        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++){
                std::cout << A_check(i,j)<<' ';
            }
            std::cout<<std::endl;
        }
        std::cout << std::endl;
    }
    void analyse(){
        Positive_definite_symmetric_matrix_generator();
        Print_Matrix();
        auto start = std::chrono::high_resolution_clock::now();
        Cholesky_dec();
        auto end = std::chrono::high_resolution_clock::now();
        Print_Lower();
        Decomposition_check();
        Print_Check_Matrix();
        printf("Chebyshev's norm =  %.10f in aij = %.10f\n", norm_max, aij_with_norm_max);
        printf("The maximum error is %.40f %% \n", norm_max*aij_with_norm_max/100);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Cholesky decomposition time : " << duration << " microseconds." << std::endl;

    }

};


int main() {
    Cholesky test(100);
    test.analyse();
    
    return 0;
}

