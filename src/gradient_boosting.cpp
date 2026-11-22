// Capstone: Gradient Boosting Histogram
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <random>
#include <cstdio>
#include <array>

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

// PROBLEM STATEMENT
// For every feature 'f' and bin 'b', sum all grad/hess of rows in 1 node if in bin 'b'

// INPUT:
// - bins[row][feature] -> bin index change with features
// - grad[row] -> fixed whatever node is
// - hess[row] -> fixed whatever node is

// OUTPUT:
// - flat global histogram array
// - attribute: hist.grad, hist.hess, hist.cut, hist.offset


struct Hist {
    vector<double> grad;
    vector<double> hess;
};


void check_hist(vector<double>& grad, vector<double>& hess, Hist& hist, int N, int F, int B) {
    double gsum = 0, hsum = 0;
    for (int r = 0; r < N; r++) { gsum += grad[r]; hsum += hess[r]; }

    bool ok = true;
    for (int f = 0; f < F; f++) {
        double fg = 0, fh = 0;
        for (int b = 0; b < B; b++) { fg += hist.grad[f * B + b]; fh += hist.hess[f * B + b]; }
        if (abs(fg - gsum) > 1e-6 * N || abs(fh - hsum) > 1e-6 * N) ok = false;
    }
    cout << "check: " << (ok ? "yes" : "NO") << endl;

}

// VERSION 0: SERIAL
void create_hist_serial(vector<int>& bin, vector<double>& grad, vector<double>& hess, Hist& hist, int N, int F, int B) {
    // Loop over row index?
    for (int n=0; n<N; n++) {
        // Extract input bin index of the row
        // R0-F0 | R0-F1 | R0-F2 | R1-F0 | R1-F1 | R1-F2        
        for (int f=0; f<F; f++) {
            int b = bin[F*n + f];

            // Find output flat index of histogram
            // F0-B0 | F0-B1 | F0-B2 | F1-B0 | F1-B1 | F1-B2
            int out = f*B + b;

            // Append grad/hess from data to that histogram index
            hist.grad[out] += grad[n];
            hist.hess[out] += hess[n];
        }
    }
}

// VERSION 1: USE LOCAL HISTOGRAM (LIKE ADVANCED HIST SESSION)
// Motivation: 
// - If using parallel over rows, multiple rows can hit the same (feature, bin) race conditions
// - Idea from histogram session: building by local histogram, then reduce
void create_hist_local(const vector<int>& bin, const vector<double>& grad,
                       const vector<double>& hess, Hist& hist, int N, int F, int B) {
    #pragma omp parallel
    {
        // Each thread has a local grad/hess parallel to hist.grad/hess
        vector<double> lg(F * B, 0.0), lh(F * B, 0.0);   // private per thread

        #pragma omp for schedule(static) nowait
        for (size_t n = 0; n < N; n++) {
            double g = grad[n], h = hess[n];             // load once per row
            for (int f = 0; f < F; f++) {
                int out = f * B + bin[n * F + f];
                lg[out] += g;
                lh[out] += h;
            }
        }

        #pragma omp critical
        for (int i = 0; i < F * B; i++) {
            hist.grad[i] += lg[i];
            hist.hess[i] += lh[i];
        }
    }
}

// VERSION 2: grad/hess of a row sitting next to each other
struct GH { double g, h; };

void create_hist_gh(const vector<int>& bin, const vector<double>& grad,
                    const vector<double>& hess, Hist& hist, int N, int F, int B) {
    #pragma omp parallel
    {
        vector<GH> lgh(F * B, {0.0, 0.0});            // private, grad+hess interleaved

        #pragma omp for schedule(static) nowait
        for (size_t n = 0; n < N; n++) {
            double g = grad[n], h = hess[n];
            for (int f = 0; f < F; f++) {
                GH& slot = lgh[f * B + bin[n * F + f]];
                // Now grad and hess entry next to each other, ensure cacheline utilized
                // F0-B0-G | F0-B0-H | F0-B1-G | F0-B1-H ...
                slot.g += g;
                slot.h += h;
            }
        }

        #pragma omp critical
        for (int i = 0; i < F * B; i++) {
            hist.grad[i] += lgh[i].g;                 // output layout unchanged
            hist.hess[i] += lgh[i].h;
        }
    }
}

// VERSION 3: PARALLEL OVER FEATURES INSTEAD OF ROWS
void create_hist_feat(const vector<int>& bin_col, const vector<double>& grad,
                      const vector<double>& hess, Hist& hist, int N, int F, int B) {
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        vector<GH> loc(B, {0.0, 0.0});               // one feature: 4 KB, stays in L1
        const int* col = &bin_col[size_t(f) * N];

        for (size_t n = 0; n < N; n++) {
            GH& s = loc[col[n]];
            s.g += grad[n];
            s.h += hess[n];
        }

        for (int b = 0; b < B; b++) {                // only this thread writes feature f
            hist.grad[f * B + b] = loc[b].g;
            hist.hess[f * B + b] = loc[b].h;
        }
    }
}

int main() {
    mt19937 rng(42);
    int N = 10000000;
    int F = 50;
    int B = 256;

    // Initialize histogram
    Hist hist;
    hist.grad.assign(F * B, 0.0);   // resize + set every element to 0
    hist.hess.assign(F * B, 0.0);

    // Offset where feature f's bin start
    vector<int> offset(F + 1);
    for (int f = 0; f <= F; f++) offset[f] = f * B;
    int total_bins = F * B;

    // Bin information of (row, feature)
    vector<int> bin(N*F, 0);
    uniform_int_distribution<int> bdist(0, B - 1);
    for (int i = 0; i < N * F; i++) bin[i] = bdist(rng);

    // Create random grad array
    vector<double> grad(N, 0.0);
    normal_distribution<double> gdist(0.0, 1.0);
    for (int r = 0; r < N; r++) grad[r] = gdist(rng);

    // Create random hess array
    vector<double> hess(N, 0.0);
    uniform_real_distribution<double> hdist(0.01, 0.25);
    for (int r = 0; r < N; r++) hess[r] = hdist(rng);

    // Column-major: bin_col[f * N + n], each feature's column is contiguous
    vector<int> bin_col(size_t(N) * F);
    for (size_t n = 0; n < N; n++)
        for (int f = 0; f < F; f++)
            bin_col[size_t(f) * N + n] = bin[n * F + f];

    // Quick check
    cout << "N=" << N << " F=" << F << " B=" << B
         << " total_bins=" << total_bins << endl;

    
    for (int i=0; i<2; i++) {
        cout << "Round: " << i << endl;
        // VERSION 0
        {
            ScopedTimer t{"hist_serial"};
            create_hist_serial(bin, grad, hess, hist, N, F, B);

        }
        // Quick check
        check_hist(grad, hess, hist, N, F, B);\
        hist.grad.assign(F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(F * B, 0.0);

        // VERSION 1
        {
            ScopedTimer t{"hist_local"};
            create_hist_local(bin, grad, hess, hist, N, F, B);

        }
        // Quick check
        check_hist(grad, hess, hist, N, F, B);\
        hist.grad.assign(F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(F * B, 0.0);

        // VERSION 2
        {
            ScopedTimer t{"hist_gh"};
            create_hist_gh(bin, grad, hess, hist, N, F, B);

        }
        // Quick check
        check_hist(grad, hess, hist, N, F, B);
        hist.grad.assign(F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(F * B, 0.0);

        // VERSION 3
        {
            ScopedTimer t{"hist_feat"};
            create_hist_feat(bin_col, grad, hess, hist, N, F, B);

        }
        // Quick check
        check_hist(grad, hess, hist, N, F, B);\
        hist.grad.assign(F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(F * B, 0.0);
    }

    return 0;
}