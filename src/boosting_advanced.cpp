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
// For every feature 'f' and bin 'b', sum all grad/hess of rows in all nodes if in bin 'b'

// INPUT:
// - bins[row][feature] -> bin index change with features
// - grad[row] -> fixed whatever node is
// - hess[row] -> fixed whatever node is
// - position[row] -> which node this row belongs to currently

// OUTPUT:
// - flat global histogram array
// - attribute: hist.grad, hist.hess, hist.cut, hist.offset

struct Hist {
    vector<double> grad;
    vector<double> hess;
};

// VERSION 0: SERIAL
void advanced_hist_serial(vector<int>& bin, vector<double>& grad, vector<double>& hess,
    vector<int>& position, Hist& hist, int N, int F, int K, int B) {
    // Histogram Layout: 
    // n0-f0-b0 … n0-f0-b255 | n0-f1-b0 … n0-f1-b255 | … | n0-f49-b255 | n1-f0-b0 …
    // └─── feature 0 ───┘   └─── feature 1 ───┘             │
    // └──────────────────── node 0: F*B = 12,800 bins ──────┘ └── node 1 ...
    // Bin Layout:
    // r0-f0 .. r0-f49 | r1-f0 .. r1-f49 | r1000-f0 .. r1000-f49

    // Mapping:
    // - (row n, feature f) -> bin info at (F * n + f)
    // - (node k, feature f, bin b) -> hist start at (k*F*B + f*B + b)

    // Loop over rows
    for (int n=0; n<N; n++) {
        // Loop over feature
        for (int f=0; f<F; f++) {
            // Access node information
            int k = position[n];

            // Access bin information
            int b = bin[F * n + f];

            // Evaluate output flat index of histogram
            int out = k * F * B + f * B + b;

            // Append grad/hess to correct position
            hist.grad[out] += grad[n];
            hist.hess[out] += hess[n];
        }
    }
}

// VERSION 0: PARALLEL OVER ROWS (LOCAL HIST)
// Motivation: 
// - If using parallel over rows, multiple rows can hit the same (feature, bin) race conditions
// - Idea from histogram session: building by local histogram, then reduce

void advanced_hist_local(vector<int>& bin, vector<double>& grad, vector<double>& hess,
    vector<int>& position, Hist& hist, int N, int F, int K, int B) {
    // Loop over rows
    #pragma omp parallel 
    {
        // Create local histogram storage 
        vector<double> lg(K * F * B), lh(K * F * B);
        #pragma omp for schedule(static) nowait
        for (int n=0; n<N; n++) {
            // Loop over feature
            for (int f=0; f<F; f++) {
                // Access node information
                int k = position[n];

                // Access bin information
                int b = bin[F * n + f];

                // Evaluate output flat index of histogram
                int out = k * F * B + f * B + b;

                // Append grad/hess to correct position
                lg[out] += grad[n];
                lh[out] += hess[n];
            }
        }

        // Critical region adding local to global histogram 1-by-1
        #pragma omp critical
        for (int i=0; i<K*F*B; i++) {
            hist.grad[i] += lg[i];
            hist.hess[i] += lh[i];
        }
    }    
}

// VERSION 3: PARALLEL OVER FEATURES INSTEAD OF ROWS
struct GH { double g, h; };
void advanced_hist_feat(const vector<int>& bin_col, const vector<double>& grad,
                      const vector<double>& hess, const vector<int>& position, Hist& hist, int N, int F, int K, int B) {
    // Invariants:
    // - input: grad/hess at row n
    // - output: get information of bin, node -> get information of flat index -> write to flat index of histogram

    // Bin Layout:
    // r0-f0 .. r1000-f0 | r0-f1 .. f1000-f1 | .. | r0-f49 .. r1000-f49
    // Mapping:
    // - (row n, feature f) -> bin info at (N * f + n)
    // - (node k, feature f, bin b) -> hist start at (k*F*B + f*B + b)

    // Process
    // - col -> r0-f20 .. r1000-f20
    // - col[10] -> r10-f20 (bin information)

    // Design:
    // - one dimension (F) is fixed per thread, so the local buffer and the global different shapes
    // - global:  hist[k][f][b]     3D, order (k, f, b)     → (k*F + f)*B + b
    // - local:   loc [k][b]        same thing with f fixed → k*B + b
    // Local storage is only K*B compared to F*K*B of previous local version

    #pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        vector<GH> loc(K * B, {0.0, 0.0});               // each feature has K*B bins
        const int* col = &bin_col[size_t(f) * N];

        // Access grad/hess at row n
        for (size_t n = 0; n < N; n++) {

            // get bin information
            int b = col[n];
            // get node information
            int k = position[n];

            // write to local flat index: (node k, bin b) -> (B*K + b)
            GH& s = loc[B * k + b];

            s.g += grad[n];
            s.h += hess[n];
        }

    // Histogram Layout: 
    // n0-f0-b0 … n0-f0-b255 | n0-f1-b0 … n0-f1-b255 | … | n0-f49-b255 | n1-f0-b0 …
    // └─── feature 0 ───┘   └─── feature 1 ───┘             │
    // └──────────────────── node 0: F*B = 12,800 bins ──────┘ └── node 1 ...

        // Get information of node 
        for (int k=0; k < K; k++) {         // each thread for a feature, disjoint writing
            for (int b = 0; b < B; b++) {                 
                // input index
                int in = B * k + b;
                int out = k * F * B + f * B + b;  
                hist.grad[out] = loc[in].g;
                hist.hess[out] = loc[in].h;
            }
        }

    }
}


void check_hist(const vector<double>& grad, const vector<double>& hess,
                const vector<int>& position, const Hist& hist,
                int N, int F, int K, int B) {
    // Expected totals per node: sum of grad/hess over the rows in that node
    vector<double> gsum(K, 0.0), hsum(K, 0.0);
    vector<long> cnt(K, 0);
    for (int r = 0; r < N; r++) {
        int k = position[r];
        if (k < 0) continue;                  // row not in an active node
        gsum[k] += grad[r];
        hsum[k] += hess[r];
        cnt[k]++;
    }

    // Every row of node k lands in exactly one bin of each feature,
    // so for each (node, feature): sum over bins == node total
    bool ok = true;
    for (int k = 0; k < K && ok; k++) {
        double tol = 1e-6 * max(1L, cnt[k]);  // tolerance scales with node size
        for (int f = 0; f < F && ok; f++) {
            double fg = 0, fh = 0;
            size_t base = (size_t(k) * F + f) * B;
            for (int b = 0; b < B; b++) { fg += hist.grad[base + b]; fh += hist.hess[base + b]; }
            if (abs(fg - gsum[k]) > tol || abs(fh - hsum[k]) > tol) {
                ok = false;
                printf("  mismatch at node %d feature %d: grad %.6f vs %.6f\n",
                       k, f, fg, gsum[k]);
            }
        }
    }
    cout << "check: " << (ok ? "yes" : "NO") << endl;
}

int main() {
    mt19937 rng(42);
    int N = 10000000;
    int F = 50;
    int B = 256;

    // Initialize position information
    int K = 64;                                     // active nodes (depth 5)
    vector<int> position(N);
    uniform_int_distribution<int> pdist(0, K - 1);
    for (int r = 0; r < N; r++) position[r] = pdist(rng);

    // Initialize histogram
    Hist hist;
    hist.grad.assign(K * F * B, 0.0);   // resize + set every element to 0
    hist.hess.assign(K * F * B, 0.0);

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
    int total_bins = K * F * B;
    cout << "N=" << N << " F=" << F << " K=" << K << " B=" << B
         << " total_bins=" << total_bins << endl;

    // VERSION 0: serial
    for (int i=0; i<2; i++) {
        cout << "Round: " << i << endl;
        // VERSION 0
        {
            ScopedTimer t{"hist_serial"};
            advanced_hist_serial(bin, grad, hess, position, hist, N, F, K, B);

        }
        // Quick check
        check_hist(grad, hess, position, hist, N, F, K, B);
        hist.grad.assign(K * F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(K * F * B, 0.0);

        // VERSION 1
        {
            ScopedTimer t{"hist_local"};
            advanced_hist_local(bin, grad, hess, position, hist, N, F, K, B);

        }
        // Quick check
        check_hist(grad, hess, position, hist, N, F, K, B);
        hist.grad.assign(K * F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(K * F * B, 0.0);

        // VERSION 2
        {
            ScopedTimer t{"hist_feat"};
            advanced_hist_feat(bin, grad, hess, position, hist, N, F, K, B);

        }
        // Quick check
        check_hist(grad, hess, position, hist, N, F, K, B);
        hist.grad.assign(K * F * B, 0.0);   // resize + set every element to 0
        hist.hess.assign(K * F * B, 0.0);
    }

    return 0;
}
