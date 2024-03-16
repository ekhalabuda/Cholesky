#include "Cholesky.h"
// Description
//
// Results:
// Base algo
// N = 100 : 790 microseconds.
// Chebyshev's norm =  0.0000000000 in aij = 4679.7271500924
// The maximum error is 0.0000000000425618704875199223603516084370 % 
// 
//
//
// Block method
// 
// D(-1) = inverse matrix D
// L(D) / LD  = lower triangular matrix for D
// B' = transposed matrix
//
//  |D  B'|
//  |B  C |  -- sourse matrix
// S = C - B'D(-1)B 
//       |
//       |
//       \/
//     | L(D)        0    |
// L = | BL(-1)(D)   L(s) |
void Cholesky::Cholesky_dec_block(){
    //split matrix A into blocks
    matrix D(d_size);
    matrix LD(d_size);
    matrix C(c_size);
    matrix S(c_size);
    FP* B = new FP[d_size*c_size];
    FP* BD = new FP[d_size*c_size];
    FP* BLD = new FP[d_size*c_size];
    size_t tmp_C = 0;
    size_t tmp_B = d_size*(d_size + 1)/2;
    size_t tmp_LA = tmp_B;
    std::memcpy(D.m, A.m, tmp_B * sizeof(FP));
    for(size_t i = 0; i < c_size; i++){
        tmp_C += i;
        std::memcpy(B + i * d_size, A.m + tmp_B, d_size * sizeof(FP));
        std::memcpy(C.m + tmp_C, A.m + d_size*(d_size + 1)/2 + d_size*(i+1) + tmp_C, (i + 1) * sizeof(FP));
        tmp_B += d_size + i + 1;
    }
    // end of spliting
    Cholesky_dec(D,LD); // LD
    std::memcpy(LA.m, LD.m, (d_size*(d_size+1)/2)*sizeof(FP));
    //LD.Print();
    inverse_lower(LD); // LD(-1)
    //LD.Print();
    //Print_matrix(B);
    multiplication_lower(B, LD, BLD); // B*LD(-1)
    //Print_matrix(BLD);
    for(size_t i = 0; i < c_size; i++){
        std::memcpy(LA.m + tmp_LA + i*(d_size), BLD + i*d_size, d_size * sizeof(FP));
        tmp_LA += i + 1;
    }
    inverse_matrix(D); // D(-1)
    multiplication_matrix(B, D, BD); // B*D(-1)
    multiplication_and_subtraction_matrix(BD, B, C); // C - B'D(-1)B 
    Cholesky_dec(C,C); // L(C - B'D(-1)B)
    tmp_LA = (d_size*(d_size+1)/2) + d_size;
    tmp_C = 0;
    for(size_t i = 0; i < c_size; i++){
        std::memcpy(LA.m + tmp_LA + i*(d_size), C.m + tmp_C, (i+1) * sizeof(FP));
        tmp_LA += i + 1;
        tmp_C += i + 1;
    }
    Print_symmetric_matrix(A);
    LA.Print();

    delete[] B;
    delete[] BD;
    delete[] BLD;
}

void Cholesky::multiplication_matrix(FP* B, matrix& M, FP* dest){
    size_t i,j,k;
    size_t rows = c_size;
    size_t cols = d_size;
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            for(k = 0; k < j; k++){
                dest[i*cols + j] += B[i*cols + k] * M(k,j);
            }
            for(k = j; k < cols; k++){
                dest[i*cols + j] += B[i*cols + k] * M(j,k);
            }
        }
    }
}

void Cholesky::multiplication_and_subtraction_matrix(FP* A, FP* B, matrix& dest){
    size_t i,j,k;
    for(i = 0; i < c_size; i++){
        for(j = 0; j <= i; j++){
            for(k = 0; k < d_size; k++){
                dest(j, i) -= A[i*d_size + k] * B[j*d_size + k];
            }

        }
    }
}

void Cholesky::multiplication_lower(FP* B, matrix& M, FP* dest){
    size_t i,j,k;
    size_t rows = c_size;
    size_t cols = d_size;
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            for(k = j; k < cols; k++){
                dest[i*cols + j] += B[i*cols + k] * M(j,k);
            }
        }
    }
}

void Cholesky::inverse_matrix(matrix& M){
    size_t n = M.matrix_size;
    size_t i, j, k;
    size_t tmp = 0;
    FP koef;
    FP* inverse = new FP[n*n*2];
    for(i = 0; i < n; i++){
        std::memcpy(inverse + i * 2 * n, M.m + tmp, (i + 1) * sizeof(FP));
        tmp += i + 1;
        inverse[i*2*n + n + i] = 1;
    }
    for(i = 0; i < n; i++){
        for(j = i; j < n; j++){
            inverse[i*2*n + j] = inverse[j*2*n + i];
        }
    }
    for(i = 0; i < n; i++){
        koef = inverse[i*2*n + i];
        for(j = 0; j <= n + i; j++){
            inverse[i*2*n + j] /= koef;
        }
        for(j = 0; j < i; j++){
            koef = inverse[j*2*n + i];
            for(k = 0; k < 2*n; k++){
                inverse[j*2*n + k] -= koef*inverse[i*2*n + k];
            }
        }
        for(j = i + 1; j < n; j++){
            koef = inverse[j*2*n + i];
            for(k = 0; k < 2*n; k++){
                inverse[j*2*n + k] -= koef*inverse[i*2*n + k];
            }
        }
    }
    tmp = 0;
    for(i = 0; i < n; i++){
        std::memcpy(M.m + tmp, inverse + i * 2 * n + n, (i + 1) * sizeof(FP) );
        tmp += i + 1;
    }
    delete[] inverse;
}

void Cholesky::inverse_lower(matrix& M){
    size_t n = M.matrix_size;
    size_t i, j, k;
    size_t tmp = 0;
    FP koef;
    FP* inverse = new FP[n*n*2];
    for(i = 0; i < n; i++){
        std::memcpy(inverse + i * 2 * n, M.m + tmp, (i + 1) * sizeof(FP));
        tmp += i + 1;
        inverse[i*2*n + n + i] = 1;
    }
    for(i = 0; i < n; i++){
        koef = inverse[i*2*n + i];
        for(j = 0; j <= n + i; j++){
            inverse[i*2*n + j] /= koef;
        }
        for(j = i + 1; j < n; j++){
            koef = inverse[j*2*n + i];
            for(k = 0; k < 2*n; k++){
                inverse[j*2*n + k] -= koef*inverse[i*2*n + k];
            }
        }
    }
    tmp = 0;
    for(i = 0; i < n; i++){
        std::memcpy(M.m + tmp, inverse + i * 2 * n + n, (i + 1) * sizeof(FP) );
        tmp += i + 1;
    }
    delete[] inverse;
}

void Cholesky::print(FP* B,size_t n){
    size_t i,j;
    for (i = 0; i < n; i++) {
        for(j = 0; j < 2*n; j++)
            std::cout << B[i*2*n+j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_matrix(FP* B){
    std::cout << "??????:" << std::endl;
    size_t i,j;
    for (i = 0; i < c_size; i++) {
        for(j = 0; j < d_size; j++)
            printf(" %.11f ", B[i*d_size+j]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_trans_matrix(FP* B){
    std::cout << "??????:" << std::endl;
    size_t i,j;
    for (i = 0; i < d_size; i++) {
        for(j = 0; j < c_size; j++)
            printf(" %.11f ", B[j*d_size+i]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Cholesky_dec(matrix& M, matrix& L) {
    L(0,0) = sqrt(M(0,0));
    size_t i, j, p;
    for(j = 1; j < M.matrix_size; j++){
        i = 0; 
        L(i,j) = M(i,j) / L(0,0);
    }
    for (i = 1; i < M.matrix_size; i++){
        FP sum = 0;
        for (p = 0; p < i; p++){
            sum += (L(p,i) * L(p,i));
        }
        L(i,i) = sqrt(std::abs(M(i,i) - sum));
        for (j = i + 1; j < M.matrix_size; j++){
            sum = 0;
            for (p = 0; p < i; p++){
                sum += L(p,i) * L(p,j);
            }
            L(i,j) = (M(i,j) - sum) / L(i,i);
        }
    }
}

void Cholesky::Positive_definite_symmetric_matrix_generator(){
    unsigned int seed = 123;
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<FP> distribution(-100.0, 100.0);
    size_t i,j;
    for(i = 0; i < N; i++)
        for(j = 0; j < i; j++){
            A(j,i) = distribution(generator);
        }

    for(i = 0; i < N; i++){
        FP s = 0; 
        for(j = 0; j < i; j++){
                s += std::abs(A(j, i));
        }
        for(j = i + 1; j < N; j++){
                s += std::abs(A(i, j));
        }
        std::uniform_real_distribution<FP> distribution_2(s, 1000.0);
        A(i,i) = distribution_2(generator);
    }
}

void Cholesky::Decomposition_check(){
    size_t i,j,k;
    for(i = 0; i < N; i++){
        for(j = 0; j <= i; j++){
            FP aij = 0;
            for(k = 0; k <= j; k++){
                aij += LA(k,i) * LA(k,j);
            }
            A_check(j,i) = aij;
            FP tmp = std::abs(aij - A(j,i));
            if(tmp > norm_max){
                norm_max = tmp;
                aij_with_norm_max = A(j,i);
            }
        }
    }
}

void Cholesky::Print_symmetric_matrix(matrix& L){
    std::cout << "Symmetric Matrix :" << std::endl;
    size_t i,j;
    for(i = 0; i < L.matrix_size; i++){
        for(j = 0; j < i; j++){
            std::cout << L(j, i)<<' ';
        }
        for(j = i; j < L.matrix_size; j++){
            std::cout << L(i, j)<<' ';
        }
        std::cout<<std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_lower(matrix& L){
    std::cout << "Lower Triangular Matrix (L):" << std::endl;
    size_t i, j;
    for (i = 0; i < N; i++) {
        for(j = 0; j <= i; j++)
            std::cout << L(j, i) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_check_matrix(){
    std::cout << "Check Matrix (A = LLт):" << std::endl;
    size_t i, j;
    for(i = 0; i < N; i++){
        for(j = 0; j < i; j++){
            std::cout << A_check(j, i)<<' ';
        }
        for(j = i; j < N; j++){
            std::cout << A_check(i, j)<<' ';
        }
        std::cout<<std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::analyse(){
    Positive_definite_symmetric_matrix_generator();
    auto start = std::chrono::high_resolution_clock::now();
    Cholesky_dec_block();
    auto end = std::chrono::high_resolution_clock::now();
    Decomposition_check();
    Cholesky_dec(A, LA);
    LA.Print();
    printf("_______________________________________________________________________________\n");
    std::cout << "Matrix size = " << N << std::endl;
    printf("Chebyshev's norm =  %.10f in aij = %.10f\n", norm_max, aij_with_norm_max);
    printf("The maximum error is %.40f %% \n", std::abs(norm_max*aij_with_norm_max/100));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Cholesky decomposition time : " << duration << " microseconds." << std::endl;

}



int main() {
    size_t n;
    std::cout << "Enter matrix size:";
    std::cin >> n;
    Cholesky test(n);
    test.analyse();
    return 0;
}

