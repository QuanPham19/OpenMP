// #include <omp.h>

// #include <cmath>
// #include <cstdio>
// #include <vector>

// // c[i] = a[i] + b[i]
// static void add_serial(const std::vector<double>& a, const std::vector<double>& b,
//                        std::vector<double>& c) {
//     const long n = static_cast<long>(a.size());
//     for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
// }

// static void add_parallel(const std::vector<double>& a, const std::vector<double>& b,
//                          std::vector<double>& c) {
//     const long n = static_cast<long>(a.size());
//     // Each iteration is independent (no shared writes), so no reduction needed.
//     #pragma omp parallel for schedule(static)
//     for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
// }

// // Run f several times and return the best (minimum) time in ms.
// template <typename F>
// static double best_time_ms(F f, int reps = 5) {
//     double best = 1e30;
//     for (int r = 0; r < reps; ++r) {
//         double t0 = omp_get_wtime();
//         f();
//         double t = (omp_get_wtime() - t0) * 1e3;
//         if (t < best) best = t;
//     }
//     return best;
// }

// int main() {
//     const long N = 20'000'000;  // 3 arrays x 20M doubles = ~480 MB total
//     std::vector<double> a(N), b(N), c_serial(N), c_parallel(N);

//     #pragma omp parallel for schedule(static)
//     for (long i = 0; i < N; ++i) {
//         a[i] = std::sin(static_cast<double>(i));
//         b[i] = std::cos(static_cast<double>(i));
//     }

//     // Correctness check first
//     add_serial(a, b, c_serial);
//     add_parallel(a, b, c_parallel);
//     double max_err = 0.0;
//     for (long i = 0; i < N; ++i)
//         max_err = std::fmax(max_err, std::fabs(c_serial[i] - c_parallel[i]));
//     std::printf("Max abs error serial vs parallel: %.3e (should be exactly 0)\n\n", max_err);

//     // Timing: serial baseline, then parallel with 1..max threads
//     double t_serial = best_time_ms([&] { add_serial(a, b, c_serial); });
//     std::printf("%-10s %10s %10s %12s\n", "threads", "time(ms)", "speedup", "GB/s");

//     // 3 arrays touched per element (2 reads + 1 write), 8 bytes each
//     const double bytes = 3.0 * 8.0 * static_cast<double>(N);
//     std::printf("%-10s %10.2f %10.2f %12.2f\n", "serial", t_serial, 1.0,
//                 bytes / (t_serial * 1e-3) / 1e9);

//     const int max_t = 96;
//     for (int t = 1; t <= max_t; t *= 2) {
//         omp_set_num_threads(t);
//         double ms = best_time_ms([&] { add_parallel(a, b, c_parallel); });
//         std::printf("%-10d %10.2f %10.2f %12.2f\n", t, ms, t_serial / ms,
//                     bytes / (ms * 1e-3) / 1e9);
//         if (t * 2 > max_t && t != max_t) t = max_t / 2;  // make sure max_t is tested
//     }
//     return 0;
// }

// Vector add on cache-sized arrays: repeat the same c = a + b many times so the
// data stays in each core's private L1/L2 cache instead of going to main memory.
#include <omp.h>

#include <cstdio>
#include <vector>

// Stops the compiler from noticing that repeating the same loop is pointless
// and deleting the repetitions.
static inline void clobber_memory() { asm volatile("" ::: "memory"); }

static double run_serial(const double* a, const double* b, double* c, long n, long reps) {
    double t0 = omp_get_wtime();
    for (long r = 0; r < reps; ++r) {
        for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
        clobber_memory();
    }
    return omp_get_wtime() - t0;
}

static double run_parallel(const double* a, const double* b, double* c, long n, long reps,
                           int threads) {
    double t0 = omp_get_wtime();
    // ONE parallel region around all repetitions: threads are created once,
    // not once per repetition (that overhead would dominate for small n).
    #pragma omp parallel num_threads(threads)
    {
        for (long r = 0; r < reps; ++r) {
            // schedule(static) gives each thread the SAME index range every
            // repetition, so its slice of a, b, c stays warm in its own cache.
            #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
            // implicit barrier at the end of "omp for" keeps repetitions in step
        }
    }
    return omp_get_wtime() - t0;
}

template <typename F>
static double best_of(F f, int tries = 3) {
    double best = 1e30;
    for (int k = 0; k < tries; ++k) {
        double t = f();
        if (t < best) best = t;
    }
    return best;
}

int main() {
    const int max_t = omp_get_max_threads();
    // Total elements processed per measurement, kept equal for every N so the
    // runs take similar time. 800M element-adds ~ a few hundred ms.
    const long total_work = 800'000'000L;

    std::printf("threads available: %d\n\n", max_t);
    std::printf("%10s %10s %8s %10s %10s %9s\n", "N", "arrays", "threads", "time(ms)",
                "GB/s", "speedup");

    for (long n : {4'000L, 16'000L, 64'000L, 256'000L, 1'000'000L, 20'000'000L}) {
        std::vector<double> a(n), b(n), c(n);
        for (long i = 0; i < n; ++i) {
            a[i] = 0.5 * static_cast<double>(i);
            b[i] = 1.0;
        }
        const long reps = total_work / n;
        const double bytes = 24.0 * static_cast<double>(n) * static_cast<double>(reps);
        const double kb = 24.0 * static_cast<double>(n) / 1024.0;

        const double t_serial = best_of([&] { return run_serial(a.data(), b.data(), c.data(), n, reps); });
        std::printf("%10ld %8.0fKB %8s %10.1f %10.1f %9.2f\n", n, kb, "serial",
                    t_serial * 1e3, bytes / t_serial / 1e9, 1.0);

        for (int t = 1; t <= max_t; t *= 2) {
            const double tp = best_of([&] { return run_parallel(a.data(), b.data(), c.data(), n, reps, t); });
            std::printf("%10s %10s %8d %10.1f %10.1f %9.2f\n", "", "", t, tp * 1e3,
                        bytes / tp / 1e9, t_serial / tp);
            if (t * 2 > max_t && t != max_t) t = max_t / 2;  // make sure max_t is tested
        }

        // sanity check
        for (long i = 0; i < n; ++i)
            if (c[i] != a[i] + b[i]) { std::printf("WRONG at %ld\n", i); return 1; }
        std::printf("\n");
    }
    return 0;
}