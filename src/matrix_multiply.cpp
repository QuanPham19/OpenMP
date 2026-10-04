#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <immintrin.h>


using namespace std;

// Define the timer
class ScopedTimer {
public:
    using ClockType = std::chrono::steady_clock;

    explicit ScopedTimer(const char* label)
        : label_{label}, start_{ClockType::now()} {}

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;
    auto operator=(ScopedTimer&&) -> ScopedTimer& = delete;

    ~ScopedTimer() {
        using namespace std::chrono;
        auto elapsed = ClockType::now() - start_;
        double ms = duration<double, std::milli>(elapsed).count();
        std::cout << ms << " ms  " << label_ << '\n';
    }

private:
    const char* label_{};
    const ClockType::time_point start_{};
};

void matrix_multiply(vector<double>& A, vector<double>& B, vector<double>& C, int M, int N, int P) {
    // Matrix A (MxN) times B (NxP) equals C (MxP)
    // C_(n, p) = sum_over_n[A_(m, n) * B(n, p)]

    for (auto m=0; m<M; ++m) {
        for (auto p=0; p<P; ++p) {
            int flat_C = m * P + p;
            double val = 0;
            for (auto n=0; n<N; ++n) {
                int flat_A = m * N + n;
                int flat_B = n * P + p; 
                val += A[flat_A] * B[flat_B];
            }
            C[flat_C] = val;
        }
    }

}

void matrix_multiply_row(vector<double>& A, vector<double>& B, vector<double>& C, int M, int N, int P) {
    // Matrix A (MxN) times B (NxP) equals C (MxP)
    // C_(n, p) = sum_over_n[A_(m, n) * B(n, p)]

    for (auto p=0; p<P; ++p) {
        for (auto m=0; m<M; ++m) {
            int flat_C = m * P + p;
            double val = 0;
            for (auto n=0; n<N; ++n) {
                int flat_A = m * N + n;
                int flat_B = n * P + p; 
                val += A[flat_A] * B[flat_B];
            }
            C[flat_C] = val;
        }
    }

}

void matrix_multiply_parallel_static(vector<double>& A, vector<double>& B, vector<double>& C, int M, int N, int P) {
    // Matrix A (MxN) times B (NxP) equals C (MxP)
    // C_(n, p) = sum_over_n[A_(m, n) * B(n, p)]
    #pragma omp parallel for schedule(static)
    for (auto m=0; m<M; ++m) {
        for (auto p=0; p<P; ++p) {
            int flat_C = m * P + p;
            double val = 0;
            for (auto n=0; n<N; ++n) {
                int flat_A = m * N + n;
                int flat_B = n * P + p; 
                val += A[flat_A] * B[flat_B];
            }
            C[flat_C] = val;
        }
    }

}

void matrix_multiply_advanced(const vector<double>& A, const vector<double>& B,
                         vector<double>& C, int M, int N, int P) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        double* c = &C[m * P];                      // find row m in C
        for (int p = 0; p < P; ++p) c[p] = 0.0;     // fill all row m of C with 0.0
        for (int n = 0; n < N; ++n) {               // for each row n of B
            const double a = A[m * N + n];          // collect the scale 'a' (same as naive)
            const double* b = &B[n * P];            // find row n in B
            for (int p = 0; p < P; ++p) {
                c[p] += a * b[p];                   // row m in C += a*(row n in B)
            }
        }
    }
}

void matrix_multiply_ultimate(const vector<double>& A, const vector<double>& B,
                         vector<double>& C, int M, int N, int P) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m += 4) {          // 4 rows of C at a time
        for (int r = 0; r < 4; ++r)
            for (int p = 0; p < P; ++p) C[(m+r)*P + p] = 0.0;
        for (int n = 0; n < N; ++n) {
            const double a0 = A[(m+0)*N + n], a1 = A[(m+1)*N + n],
                        a2 = A[(m+2)*N + n], a3 = A[(m+3)*N + n];
            const double* b = &B[n * P];
            for (int p = 0; p < P; ++p) {
                const double bp = b[p];             // load once... (here also prefetch)
                C[(m+0)*P + p] += a0 * bp;          // ...use 4 times
                C[(m+1)*P + p] += a1 * bp;
                C[(m+2)*P + p] += a2 * bp;
                C[(m+3)*P + p] += a3 * bp;
            }
        }
    }
}

void matmul_regblock(const vector<double>& A, const vector<double>& B,
                     vector<double>& C, int M, int N, int P) {
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < P; p += 8) {          // B panel: 8 cols x N rows ≈ 192 KB, stays in L2
        for (int m = 0; m < M; m += 4) {
            __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();

            for (int n = 0; n < N; ++n) {
                __m256d b0 = _mm256_loadu_pd(&B[n*P + p]);
                __m256d b1 = _mm256_loadu_pd(&B[n*P + p + 4]);
                __m256d a;
                a = _mm256_broadcast_sd(&A[(m+0)*N + n]);
                c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
                a = _mm256_broadcast_sd(&A[(m+1)*N + n]);
                c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
                a = _mm256_broadcast_sd(&A[(m+2)*N + n]);
                c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
                a = _mm256_broadcast_sd(&A[(m+3)*N + n]);
                c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
            }
            _mm256_storeu_pd(&C[(m+0)*P + p], c00); _mm256_storeu_pd(&C[(m+0)*P + p + 4], c01);
            _mm256_storeu_pd(&C[(m+1)*P + p], c10); _mm256_storeu_pd(&C[(m+1)*P + p + 4], c11);
            _mm256_storeu_pd(&C[(m+2)*P + p], c20); _mm256_storeu_pd(&C[(m+2)*P + p + 4], c21);
            _mm256_storeu_pd(&C[(m+3)*P + p], c30); _mm256_storeu_pd(&C[(m+3)*P + p + 4], c31);
        }
    }
}

int main() {
    int M = 2000;
    int N = 3000; 
    int P = 1000; 

    // A = [[1, 1, 1], [1, 1, 1]]
    // B = [2, 2, 2]
    vector<double> A(M*N, 1.0);
    vector<double> B(N*P, 2.0);
    vector<double> C(M*P, 0.0);

    for (auto i=0; i<3; i++) {
        cout << "round: " << i << endl;
        {
            ScopedTimer t{"naive_matrix_multiply"};
            matrix_multiply_row(A, B, C, M, N, P);
        }

        {
            ScopedTimer t{"parallel_matrix_multiply_static"};
            matrix_multiply_parallel_static(A, B, C, M, N, P);
        }

        {
            ScopedTimer t{"parallel_matrix_multiply_advanced"};
            matrix_multiply_advanced(A, B, C, M, N, P);
        }

        {
            ScopedTimer t{"parallel_matrix_multiply_ultimate"};
            matrix_multiply_ultimate(A, B, C, M, N, P);
        }

        {
            ScopedTimer t{"matmul_regblock"};
            matmul_regblock(A, B, C, M, N, P);
        }
    }
    // for (auto c: C) {cout << c << ", ";}
    return 0;
}