#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <random>
#include <cstdio>

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

void histogram_reduction(vector<int>& data, size_t N, vector<int>& hist, int B) {
    // array-section reduction needs a raw pointer
    // OpenMP only understand plain C: arrays, scalars, pointers (not std::vector)
    int* h = hist.data();   

    // each thread gets copy of h[0..B-1] and at the end reducing by (+) to the original
    // reduction (
    // + -> how to combine copies at the end
    // h[:B] -> what to be privatized)
    #pragma omp parallel for schedule(static) reduction(+ : h[:B])
    for (size_t i = 0; i < N; i++) {
        h[data[i]] += 1;
    }
}

void histogram_multi(vector<int>& data, size_t N, vector<int>& hist, int B) {
    #pragma omp parallel
    {
        // Using sub-histogram to avoid near-by elements hitting the same bin
        // Each thread has its own l0, l1, l2, l3
        vector<int> l0(B, 0), l1(B, 0), l2(B, 0), l3(B, 0);

        // Notice: it is NOT the case that we 'finish each index i' sequentially
        // - Steps in an 'i-increment': load data[i] -> load histogram counter -> add 1 -> store the new value
        // - While 'i' is in [add steps], 'i+1' can already be in [load steps]
        // - Out-of-order execution: prefetch/look-ahead data and instructions and smart-decide what to do next

        // Where does the stall of 'without histogram' come from? -> check dependency of writings
        // All the below index belong to 1 core (saying 0, 1, 2 on core-1 and 3, 4, 5 on core-2)
        // - i=0: h[1] += 1    load h[1] → add → store h[1]
        // - i=1: h[2] += 1    different address → can overlap with i=0 ✓
        // - i=2: h[1] += 1    same address as i=0 → must wait for i=0's store ✗ (inefficiency)

        // Two upgrades so far:
        // - histogram_local/reduction: each core has its own 'hist' to avoid cacheline invalidation between cores
        // - histogram_multi: each core has its own 'sub-hist' to avoid writing to same address stall in 1 core
        #pragma omp for schedule(static) nowait 
        for (auto i=0; i < N - N%4; i+=4) {
            l0[data[i]] += 1;
            l1[data[i+1]] += 1;
            l2[data[i+2]] += 1;
            l3[data[i+3]] += 1;
        }

        #pragma omp critical
        for (int b=0; b<B; b++) {
            hist[b] += (l0[b] + l1[b] + l2[b] + l3[b]);
        }
    }
    // leftover elements (0–3)
    for (size_t i = N - N % 4; i < N; i++)   
        hist[data[i]] += 1;
}

// So what if we increase the number of sub-histograms?
template <int K>
void histogram_multiK(vector<int>& data, size_t N, vector<int>& hist, int B) {
    #pragma omp parallel
    {
        vector<int> local(K * B, 0);          // K sub-histograms, back to back

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < N - N % K; i += K) {
            for (int k = 0; k < K; k++)       // K known at compile time → unrolled
                local[k * B + data[i + k]] += 1;
        }

        #pragma omp critical
        for (int b = 0; b < B; b++) {
            int s = 0;
            for (int k = 0; k < K; k++) s += local[k * B + b];
            hist[b] += s;
        }
    }
    for (size_t i = N - N % K; i < N; i++)
        hist[data[i]] += 1;
}


int main() {
    // Declare size of B and N
    const int N = 1e8;
    const int B = 256;

    // Build data and hist
    vector<int> data(N);
    mt19937 rng(42);
    uniform_int_distribution<int> dist_all(0, B - 1);   // any bin
    uniform_int_distribution<int> dist_hot(0, 2);       // bins 0, 1, 2 only
    bernoulli_distribution is_hot(0.9);                 // true 90% of the time

    for (int i = 0; i < N; i++) {
        data[i] = is_hot(rng) ? dist_hot(rng) : dist_all(rng);
    }

    vector<int> hist(B, 0);

    for (auto i=0; i<2; i++) {
        
        cout << "round: " << i << endl;
        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_reduction"};
            histogram_reduction(data, N, hist, B);
        }

        int total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
        cout << "hist[0..3] = " << hist[0] << " " << hist[1] << " " << hist[2] << " " << hist[3] << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_multi"};
            histogram_multi(data, N, hist, B);
        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
        cout << "hist[0..3] = " << hist[0] << " " << hist[1] << " " << hist[2] << " " << hist[3] << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_multi_8"};
            histogram_multiK<8>(data, N, hist, B);

        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
        cout << "hist[0..3] = " << hist[0] << " " << hist[1] << " " << hist[2] << " " << hist[3] << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_multi_16"};
            histogram_multiK<16>(data, N, hist, B);

        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
        cout << "hist[0..3] = " << hist[0] << " " << hist[1] << " " << hist[2] << " " << hist[3] << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_multi_32"};
            histogram_multiK<32>(data, N, hist, B);

        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
        cout << "hist[0..3] = " << hist[0] << " " << hist[1] << " " << hist[2] << " " << hist[3] << endl;
    }

    return 0;
}