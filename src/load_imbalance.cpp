#include <omp.h>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>

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

// EXAMPLE:
// W = 5, H = 5          image size in pixels
// MAXIT = 1000          give up after this many iterations
// region: real part  -2.0 .. 1.0
//         imag part  -1.5 .. 1.5
// cr = -2.0 + 3.0 * x / W     → x = 0..4 gives cr = -2.0, -1.4, -0.8, -0.2, 0.4
// ci = -1.5 + 3.0 * y / H     → y = 0..4 gives ci = -1.5, -0.9, -0.3,  0.3, 0.9

// WHAT ARE WE DOING?
// - Input: width, height, max_iter, region
// - From input, auto-generate the whole array of complex c: (real, imaginary) - attribute for every pixel on the grid
// - On each pixel, start z = 0 and keep iterating: z = z^2 + c until |z| > 2
// - Output: array of iterations to escape for each pixel (e.g. [1, 1000, 500, 2,...])

int mandel_pixel(double cr, double ci, int MAXIT) {
    double zr = 0.0, zi = 0.0;
    int it = 0;
    while (it < MAXIT && zr * zr + zi * zi <= 4.0) {
        double t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        it++;
    }
    return it;
}

void print_image(vector<int>& iters, int W, int H, int MAXIT) {
    // Top border
    cout << '+' << string(W, '-') << "+\n";

    for (int h = 0; h < H; h++) {
        cout << '|';                                   // left border
        for (int w = 0; w < W; w++) {
            int it = iters[h * W + w];
            cout << (it == MAXIT ? '#' : it > 10 ? '+' : it > 3 ? '.' : ' ');
        }
        cout << "|\n";                                 // right border
    }

    // Bottom border
    cout << '+' << string(W, '-') << "+\n";
}

// Flat index = h*W + w
void mandel_naive(vector<int>& iters, int W, int H, int MAXIT) {
    for (int h = 0; h < H; h++) {          // rows outside
        for (int w = 0; w < W; w++) {      // columns inside → contiguous writes
            double cr = -2.0 + 3.0 * w / W;
            double ci = -1.5 + 3.0 * h / H;
            iters[h * W + w] = mandel_pixel(cr, ci, MAXIT);
        }
    }
}

// Parallel naive
// void mandel_parallel(vector<int>& iters, int W, int H, int MAXIT) {
//     #pragma omp parallel for schedule(static)
//     for (int h = 0; h < H; h++) {          // rows outside
//         for (int w = 0; w < W; w++) {      // columns inside → contiguous writes
//             double cr = -2.0 + 3.0 * w / W;
//             double ci = -1.5 + 3.0 * h / H;
//             iters[h * W + w] = mandel_pixel(cr, ci, MAXIT);
//         }
//     }
// }
void mandel_parallel(vector<int>& iters, int W, int H, int MAXIT) {
    const int T = omp_get_max_threads();
    vector<double> busy_ms(T, 0.0);
    vector<int> rows_done(T, 0);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        double t0 = omp_get_wtime();
        int rows = 0;

        #pragma omp for schedule(static) nowait   // nowait: measure each thread's own work
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                double cr = -2.0 + 3.0 * w / W;
                double ci = -1.5 + 3.0 * h / H;
                iters[h * W + w] = mandel_pixel(cr, ci, MAXIT);
            }
            rows++;
        }

        busy_ms[t] = (omp_get_wtime() - t0) * 1e3;   // written once per thread
        rows_done[t] = rows;
    }

    double total_ms = (omp_get_wtime() - start) * 1e3;

    for (int t = 0; t < T; t++) {
        printf("  T%d: busy %7.1f ms  wait %7.1f ms  rows %d\n",
               t, busy_ms[t], total_ms - busy_ms[t], rows_done[t]);
    }
}

// schedule(static)      T0: 0–499       T1: 500–999      T2: 1000–1499    T3: 1500–1999
// schedule(static, 1)   T0: 0,4,8,...   T1: 1,5,9,...    T2: 2,6,10,...   T3: 3,7,11,...
void mandel_static(vector<int>& iters, int W, int H, int MAXIT) {
    const int T = omp_get_max_threads();
    vector<double> busy_ms(T, 0.0);
    vector<int> rows_done(T, 0);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        double t0 = omp_get_wtime();
        int rows = 0;

        #pragma omp for schedule(static, 1) nowait   // nowait: measure each thread's own work
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                double cr = -2.0 + 3.0 * w / W;
                double ci = -1.5 + 3.0 * h / H;
                iters[h * W + w] = mandel_pixel(cr, ci, MAXIT);
            }
            rows++;
        }

        busy_ms[t] = (omp_get_wtime() - t0) * 1e3;   // written once per thread
        rows_done[t] = rows;
    }

    double total_ms = (omp_get_wtime() - start) * 1e3;

    for (int t = 0; t < T; t++) {
        printf("  T%d: busy %7.1f ms  wait %7.1f ms  rows %d\n",
               t, busy_ms[t], total_ms - busy_ms[t], rows_done[t]);
    }
}

// Dynamic: 2000 rows → cut into chunks of 16 → 125 chunks in a shared queue
// T0 grabs rows 0–15      T1 grabs 16–31     T2 grabs 32–47     T3 grabs 48–63
// T0 finishes first → grabs the next free chunk (64–79)
// T3 finishes next  → grabs 80–95 ... until the queue is empty

void mandel_dynamic(vector<int>& iters, int W, int H, int MAXIT) {
    const int T = omp_get_max_threads();
    vector<double> busy_ms(T, 0.0);
    vector<int> rows_done(T, 0);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        double t0 = omp_get_wtime();
        int rows = 0;

        #pragma omp for schedule(dynamic, 16) nowait   // nowait: measure each thread's own work
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                double cr = -2.0 + 3.0 * w / W;
                double ci = -1.5 + 3.0 * h / H;
                iters[h * W + w] = mandel_pixel(cr, ci, MAXIT);
            }
            rows++;
        }

        busy_ms[t] = (omp_get_wtime() - t0) * 1e3;   // written once per thread
        rows_done[t] = rows;
    }

    double total_ms = (omp_get_wtime() - start) * 1e3;

    for (int t = 0; t < T; t++) {
        printf("  T%d: busy %7.1f ms  wait %7.1f ms  rows %d\n",
               t, busy_ms[t], total_ms - busy_ms[t], rows_done[t]);
    }
}

void mandel_fast(vector<int>& iters, int W, int H, int MAXIT) {
    const int L = 8;                                  // pixels processed together per thread

    #pragma omp parallel for schedule(dynamic, 16)
    for (int h = 0; h < H; h++) {
        double ci = -1.5 + 3.0 * h / H;
        int w0 = 0;

        for (; w0 + L <= W; w0 += L) {
            double cr[L], zr[L], zi[L];
            int it[L];

            for (int l = 0; l < L; l++) {
                cr[l] = -2.0 + 3.0 * (w0 + l) / W;
                zr[l] = 0.0; zi[l] = 0.0;
                it[l] = 0;                            // no interior shortcut: every pixel iterates
            }

            for (int k = 0; k < MAXIT; k++) {
                bool any = false;
                #pragma omp simd reduction(|| : any)
                for (int l = 0; l < L; l++) {
                    double zr2 = zr[l] * zr[l], zi2 = zi[l] * zi[l];
                    bool active = (zr2 + zi2 <= 4.0);
                    double nzr = zr2 - zi2 + cr[l];
                    double nzi = 2.0 * zr[l] * zi[l] + ci;
                    zr[l] = active ? nzr : zr[l];     // escaped lanes stay frozen
                    zi[l] = active ? nzi : zi[l];
                    it[l] += active;
                    any |= active;
                }
                if (!any) break;                      // all 8 pixels escaped
            }

            for (int l = 0; l < L; l++) iters[h * W + w0 + l] = it[l];
        }

        // Leftover columns if W is not a multiple of L
        for (; w0 < W; w0++) {
            double cr = -2.0 + 3.0 * w0 / W;
            iters[h * W + w0] = mandel_pixel(cr, ci, MAXIT);
        }
    }
}

int main() {
    int W = 2000;
    int H = 2000;
    int MAXIT = 1000;
    vector<int> iters(W*H, 0);
    for (auto i=0; i<2; ++i) {
        cout << "Round: " << i << endl;
        {
            ScopedTimer t{"mandel_naive"};
            mandel_naive(iters, W, H, MAXIT);
        }

        fill(iters.begin(), iters.end(), 0);
        {
            ScopedTimer t{"mandel_parallel"};
            mandel_parallel(iters, W, H, MAXIT);
        }

        fill(iters.begin(), iters.end(), 0);
        {
            ScopedTimer t{"mandel_static"};
            mandel_static(iters, W, H, MAXIT);
        }

        fill(iters.begin(), iters.end(), 0);
        {
            ScopedTimer t{"mandel_dynamic"};
            mandel_dynamic(iters, W, H, MAXIT);
        }

        fill(iters.begin(), iters.end(), 0);
        {
            ScopedTimer t{"mandel_fast"};
            mandel_fast(iters, W, H, MAXIT);
        }
    }

    vector<int> ref(W * H), out(W * H);
    mandel_naive(ref, W, H, MAXIT);
    mandel_fast(out, W, H, MAXIT);
    cout << (ref == out ? "match" : "MISMATCH") << endl;

    // print_image(iters, W, H, MAXIT);
}