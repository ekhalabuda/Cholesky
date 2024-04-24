#g++ -Wall -O3 -fopenmp -std=c++11 Cholesky.cpp -o Cholesky
g++ -Wall -O3 -fopenmp -std=c++11 -I/opt/intel/oneapi/mkl/2024.1/include Cholesky.cpp -L/opt/intel/oneapi/mkl/2024.1/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -o Cholesky
