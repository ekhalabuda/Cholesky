#include "Cholesky.h"
// Description
//
// Results:
// Base algo
// Matrix size = 1000
// Chebyshev's norm =  0.0000000000 in aij = 45673.6244748528
// The maximum error is 0.0000000033231935576515983784044876744694 % 
// Cholesky decomposition time : 702507 microseconds.
// _______________________________________________________________________________
// Block algo
// Matrix size = 1000
// A block size = 500
// Chebyshev's norm =  0.0000000002 in aij = 48956.1380986460
// The maximum error is 0.0000000819266407247675735386760876959034 % 
// Cholesky decomposition time : 1981307 microseconds.
// _______________________________________________________________________________
// Matrix size = 1000
// A block size = 14
// Chebyshev's norm =  0.0000000000 in aij = 46268.2689926411
// The maximum error is 0.0000000134658385628837684352311366555929 % 
// Cholesky decomposition time : 677767 microseconds.
//27.03.24
// _______________________________________________________________________________
// Matrix size = 1000
// Chebyshev's norm =  0.0000000000 in aij = 45673.6244748528
// The maximum error is 0.0000000033231935576515983784044
// Cholesky decomposition time : 743138 microseconds.
// _______________________________________________________________________________
// Matrix size = 1000
// A block size = 14
// Chebyshev's norm =  0.0000000000 in aij = 46268.2689926411
// The maximum error is 0.0000000134658385628837684352311366555929 % 
// Cholesky decomposition time : 733998 microseconds.
// _______________________________________________________________________________
// Matrix size = 1000
// A block size = 500
// Chebyshev's norm =  0.0000000002 in aij = 48956.1380986460
// The maximum error is 0.0000000819266407247675735386760876959034 % 
// Cholesky decomposition time : 1784482 microseconds.
// //
// еще 30 - 50
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
void Cholesky::Cholesky_dec_block(size_t n) {
    if (N - n <= d_size*2) {
        // block (d_size * 2)*(d_size * 2 + 1)/2
        Cholesky_dec(n, N);
        return;
    }
    //else
    c_size = N - n - d_size;
    multiplication_inverse_matrix(n);
    multiplication_and_substraction_matrix_DB(n); // S
    Cholesky_dec(n, n + d_size); // LD
    multiplication_inverse_lower(n); // BL(-1)(D) 
    Cholesky_dec_block(n + d_size); // L(s) 

}


void Cholesky::multiplication_and_substraction_matrix_DB(size_t begin){
    size_t j, k, p, h;
    size_t i = begin + d_size;//current row
    size_t number_of_blocks = (c_size - 1)/ d_size + 1;
    #pragma omp parallel for schedule(dynamic, 4) private(h, i, j, k, p)
    for(p = 0; p < number_of_blocks; p++){
        size_t len_of_block = std::min(d_size,  N - begin - d_size*(p+1));
        for(h = 0; h < p; h++){
            for(i = begin + d_size*(p+1); i < begin + d_size*(p+1) + len_of_block; i++){
                for(j = begin + d_size*(h+1); j < begin + d_size*(h+2); j++){
                    for(k = begin + d_size*(h+1); k < begin + d_size*(h+2); k++){
                        LA(j, i) -= B_copy[(i - begin - d_size) * d_size + (k - begin - d_size*(h+1))] * LA(k-d_size*(h+1), j);
                    }
                }
            }
        }
        for(i = begin + d_size*(p+1); i < begin + d_size*(p+1) + len_of_block; i++){
            for(j = begin + d_size*(p+1); j <= i; j++){
                for(k = begin + d_size*(p+1); k < begin + d_size*(p+2); k++){
                    LA(j, i) -= B_copy[(i - begin - d_size) * d_size + (k - begin - d_size*(p+1))] * LA(k-d_size*(p+1), j);
                }
            }
        }
    }

}

void Cholesky::multiplication_inverse_matrix(size_t begin){
    size_t i, j, k, p;
    size_t tmp = begin + begin*(begin+1)/2;
    FP koef, koef1;
    size_t block_size = d_size*d_size;
    size_t number_of_blocks = (c_size - 1)/ block_size + 1;
    for (i = 0; i < d_size; i++) {
        std::memcpy(inverse + i * d_size, LA.m + tmp, (i + 1) * sizeof(FP));
        tmp += i + 1 + begin;
    }  
    for (i = 0; i < d_size; i++) {
        for (j = i; j < d_size; j++) {
            inverse[i * d_size + j] = inverse[j * d_size + i];
        }
    }
    for (i = 0; i < c_size; i++){
        std::memcpy(B_copy + i * d_size, LA.m + tmp, d_size * sizeof(FP));
        tmp += d_size + i + 1 + begin;
    }

    for (i = 0; i < d_size; i++) {
        koef = inverse[i * d_size + i];
        for (j = 0; j < i; j++) {
            koef1 = inverse[j * d_size + i];
            for (k = 0; k < d_size; k++) {
                inverse[j * d_size + k] -= koef1 * inverse[i * d_size + k] / koef;
            }
            #pragma omp parallel for private(p, k)
            for (p = 0; p < number_of_blocks; p++){
                size_t len_of_block = std::min(block_size, c_size - block_size * p);
                for (k = block_size * p; k < block_size * p + len_of_block; k++){
                    B_copy[k * d_size + j] -= koef1 * B_copy[k * d_size + i] / koef;
                }
            }
        }
        for (j = i + 1; j < d_size; j++) {
            koef1 = inverse[j * d_size + i];
            for (k = 0; k < d_size; k++) {
                inverse[j * d_size + k] -= koef1 * inverse[i * d_size + k];
            }
            #pragma omp parallel for private(p, k)
            for (p = 0; p < number_of_blocks; p++){
                size_t len_of_block = std::min(block_size, c_size - block_size * p);
                for (k = block_size * p; k < block_size * p + len_of_block; k++){
                    B_copy[k * d_size + j] -= koef1 * B_copy[k * d_size + i];
                }
            }
        }

    }
}

void Cholesky::multiplication_inverse_lower(size_t begin) {
    size_t i, j, k, p;
    FP koef, koef1;
    size_t block_size = d_size*d_size;
    size_t number_of_blocks = (c_size - 1)/ block_size + 1;
    for (i = begin; i < d_size + begin; i++) {
        koef = LA(i, i);
        #pragma omp parallel for private(j,k,koef1)
        for (p = 0; p < number_of_blocks; p++){
            size_t len_of_block = std::min(block_size, N - begin - block_size*p - d_size);
            for (j = begin + block_size*p + d_size; j < begin + block_size*p+d_size + len_of_block; j++) {
                LA(i, j) /= koef;
            }
            for (j = i + 1; j < d_size + begin ; j++) {
                koef1 = LA(i, j);
                for (k = begin + block_size*p + d_size; k < begin + block_size*p+d_size + len_of_block; k++) {
                    LA(j, k) -= koef1 * LA(i, k);
                }
            }
        }
    }
}

void Cholesky::Cholesky_dec(size_t begin, size_t end) {
    LA(begin, begin) = sqrt(LA(begin, begin));
    size_t i, j, p;
    for (j = begin + 1; j < end; j++) {
        i = begin;
        LA(i, j) = LA(i, j) / LA(begin, begin);
    }
    for (i = begin + 1; i < end; i++) {
        FP sum = 0;
        for (p = begin; p < i; p++) {
            sum += (LA(p, i) * LA(p, i));
        }
        LA(i, i) = sqrt(std::abs(LA(i, i) - sum));
        for (j = i + 1; j < end; j++) {
            sum = 0;
            for (p = begin; p < i; p++) {
                sum += LA(p, i) * LA(p, j);
            }
            LA(i, j) = (LA(i, j) - sum) / LA(i, i);
        }
    }
}

void Cholesky::inverse_matrix(size_t begin) {
    size_t n = d_size;
    size_t i, j, k;
    size_t tmp = begin + begin*(begin+1)/2;
    FP koef;
    for (i = 0; i < n; i++) {
        std::memcpy(inverse + i * 2 * n, LA.m + tmp, (i + 1) * sizeof(FP));
        tmp += i + 1 + begin;
        inverse[i * 2 * n + n + i] = 1.;
    }  
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            inverse[i * 2 * n + j] = inverse[j * 2 * n + i];
        }
    }
    for (i = 0; i < n; i++) {
        koef = inverse[i * 2 * n + i];
        for (j = 0; j <= n + i; j++) {
            inverse[i * 2 * n + j] /= koef;
        }
        for (j = 0; j < i; j++) {
            koef = inverse[j * 2 * n + i];
            for (k = 0; k < 2 * n; k++) {
                inverse[j * 2 * n + k] -= koef * inverse[i * 2 * n + k];
            }
        }
        for (j = i + 1; j < n; j++) {
            koef = inverse[j * 2 * n + i];
            for (k = 0; k < 2 * n; k++) {
                inverse[j * 2 * n + k] -= koef * inverse[i * 2 * n + k];
            }
        }
    }
    tmp = 0;
    for(i = 0; i < n; i++){
        std::memcpy(D.m + tmp, inverse + i * 2 * n + n, (i + 1) * sizeof(FP) );
        tmp += i + 1;
    }
    std::memset(inverse, 0.0, d_size * d_size * 2 * sizeof(FP));
}

void Cholesky::inverse_lower(size_t begin) {
    size_t i, j, k;
    FP koef;
    std::memset(D.m, 0, d_size * (d_size + 1) / 2 * sizeof(FP));
    for (i = 0; i < d_size; i++) {
        koef = LA(i + begin, i + begin);
        for (j = 0; j < i; j++) {
            D(j, i) /= koef;
        }
        D(j, i) = 1.0 / koef;
        for (j = i + 1; j < d_size ; j++) {
            koef = LA(i + begin, j + begin);
            for (k = 0; k <= i; k++) {
                D(k, j) -= koef * D(k, i);
            }
        }
    }
}

void Cholesky::print(const FP* B) {
    size_t i, j;
    for (i = 0; i < c_size; i++) {
        for (j = 0; j < d_size; j++)
            std::cout << B[i *d_size + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_matrix(const FP* B) {
    std::cout << "??????:" << std::endl;
    size_t i, j;
    for (i = 0; i < c_size; i++) {
        for (j = 0; j < d_size; j++)
            printf(" %.11f ", B[i * d_size + j]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_trans_matrix(const FP* B) {
    std::cout << "??????:" << std::endl;
    size_t i, j;
    for (i = 0; i < d_size; i++) {
        for (j = 0; j < c_size; j++)
            printf(" %.11f ", B[j * d_size + i]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_symmetric_matrix(const matrix& L) {
    std::cout << "Symmetric Matrix :" << std::endl;
    size_t i, j;
    for (i = 0; i < L.matrix_size; i++) {
        for (j = 0; j < i; j++) {
            std::cout << L(j, i) << " ";
        }
        for (j = i; j < L.matrix_size; j++) {
            std::cout << L(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_lower(const matrix& L) {
    std::cout << "Lower Triangular Matrix (L):" << std::endl;
    size_t i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++)
            std::cout << L(j, i) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Print_check_matrix() {
    std::cout << "Check Matrix (A = LLт):" << std::endl;
    size_t i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {
            std::cout << A_check(j, i) << ' ';
        }
        for (j = i; j < N; j++) {
            std::cout << A_check(i, j) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Cholesky::Positive_definite_symmetric_matrix_generator() {
    unsigned int seed = 123;
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<FP> distribution(-100.0, 100.0);
    size_t i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < i; j++) {
            A(j, i) = distribution(generator);
            LA(j,i) = A(j,i);
        }

    for (i = 0; i < N; i++) {
        FP s = 0;
        for (j = 0; j < i; j++) {
            s += std::abs(A(j, i));
        }
        for (j = i + 1; j < N; j++) {
            s += std::abs(A(i, j));
        }
        std::uniform_real_distribution<FP> distribution_2(1.0, 1000.0);
        A(i, i) = s + distribution_2(generator);
        LA(i, i) = A(i, i);

    }
}

void Cholesky::Decomposition_check() {
    size_t i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
            FP aij = 0;
            for (k = 0; k <= j; k++) {
                aij += LA(k, i) * LA(k, j);
            }
            A_check(j, i) = aij;
            FP tmp = std::abs(aij - A(j, i));
            if (tmp > norm_max) {
                norm_max = tmp;
                aij_with_norm_max = A(j, i);
            }
        }
    }
}

void Cholesky::analyse() {
    Positive_definite_symmetric_matrix_generator();
    //Print_symmetric_matrix(A);
    auto start = std::chrono::high_resolution_clock::now();
    //Cholesky_dec(0, N);
    Cholesky_dec_block(0);
    auto end = std::chrono::high_resolution_clock::now();
    //Print_symmetric_matrix(A);
    //Print_lower(LA);
    //Decomposition_check();
    //Print_check_matrix();
    printf("_______________________________________________________________________________\n");
    std::cout << "Matrix size = " << N << std::endl;
    std::cout << "A block size = " << d_size << std::endl;
    printf("Chebyshev's norm =  %.10f in aij = %.10f\n", norm_max, aij_with_norm_max);
    printf("The maximum error is %.40f %% \n", std::abs(norm_max * aij_with_norm_max / 100));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Cholesky decomposition time : " << duration << " microseconds." << std::endl;
 
}



void test(){
    Cholesky test1(5000, 5000);
    test1.analyse();
    for(size_t i = 100; i <= 160; i++){
        Cholesky test(5000, i);
        test.analyse();
    }
}

int main() {
    //test();
    size_t n;
    std::cout << "Enter matrix size:";
    std::cin >> n;
    size_t num;
    std::cout << "Enter block size:";
    std::cin >> num;
    Cholesky test(n, num);
    test.analyse();
    return 0;
}

 
