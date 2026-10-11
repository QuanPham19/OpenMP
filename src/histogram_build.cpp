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

void histogram_naive(vector<int>& data, size_t N, vector<int>& hist, int B) {
    // Context: building histogram
    // Array [data] of size N
    // All values within [0 .. B-1]
    // Histogram [hist] satisfying that hist[b] = number of elements equal to B
    for (auto b: data) {
        hist[b] += 1;
    }
}

void histogram_race(vector<int>& data, size_t N, vector<int>& hist, int B) {
    // Naive race parallelism: multiple core/thread writing to the same bins
    // Output is incorect, and even slower by 10x (due to cache invalidation effect)
    #pragma omp parallel for schedule(static)
    for (auto b: data) {
        hist[b] += 1;
    }
}

void histogram_local(vector<int>& data, size_t N, vector<int>& hist, int B) {
    #pragma omp parallel 
    {
        // Each thread has a private histogram (on its own cache)
        // No more cache invalidation issue, since everything is privatized
        vector<int> local(B, 0);

        // Split data across threads, each only writes to own local
        // Nowait useful when later works independent of the current for loop
        #pragma omp for schedule(static) nowait
        for (auto i=0; i<N; i++) {
            local[data[i]] += 1;
        }

        // Merge locals to global by cheap lock
        #pragma omp critical
        for (auto b=0; b<B; b++) {
            hist[b] += local[b];
        }

    }
}

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

int main() {
    // Declare size of B and N
    const int N = 1e8;
    const int B = 256;

    // Build data and hist
    vector<int> data(N);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, B-1);
    for (auto i=0; i<N; i++) {
        data[i] = dist(rng);
    }
    vector<int> hist(B, 0);

    for (auto i=0; i<2; i++) {
        
        cout << "round: " << i << endl;
        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_naive"};
            histogram_naive(data, N, hist, B);
        }

        long long total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_race"};
            histogram_race(data, N, hist, B);
        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_local"};
            histogram_local(data, N, hist, B);
        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;

        fill(hist.begin(), hist.end(), 0);
        {
            ScopedTimer t{"histogram_reduction"};
            histogram_reduction(data, N, hist, B);
        }

        total = 0;
        for (int b = 0; b < B; b++) total += hist[b];
        cout << "Sanity check: " << total << endl;
    }

    return 0;
}