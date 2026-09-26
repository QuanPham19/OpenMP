#include <omp.h>

#include <cmath>
#include <cstdio>
#include <vector>

// Some non-trivial work per element so threads have something to chew on.
static double work(double x) {
    return std::sin(x) * std::cos(x) + std::sqrt(x);
}

int main() {
    // 1) How many threads do we have?
    std::printf("Max threads available: %d\n", omp_get_max_threads());

    // 2) Hello from each thread
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n = omp_get_num_threads();
        #pragma omp critical
        std::printf("  hello from thread %d of %d\n", tid, n);
    }

    // 3) Serial vs parallel sum (reduction)
    const long N = 50'000'000;
    std::vector<double> data(N);
    for (long i = 0; i < N; ++i) data[i] = static_cast<double>(i % 1000) + 1.0;

    double t0 = omp_get_wtime();
    double serial_sum = 0.0;
    for (long i = 0; i < N; ++i) serial_sum += work(data[i]);
    double t_serial = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    double parallel_sum = 0.0;
    #pragma omp parallel for reduction(+ : parallel_sum) schedule(static)
    for (long i = 0; i < N; ++i) parallel_sum += work(data[i]);
    double t_parallel = omp_get_wtime() - t0;

    std::printf("\nSerial   sum = %.6e  time = %.3f s\n", serial_sum, t_serial);
    std::printf("Parallel sum = %.6e  time = %.3f s\n", parallel_sum, t_parallel);
    std::printf("Speedup: %.2fx\n", t_serial / t_parallel);
    std::printf("Relative diff: %.2e  (small FP differences are expected)\n",
                std::fabs(serial_sum - parallel_sum) / std::fabs(serial_sum));
    return 0;
}