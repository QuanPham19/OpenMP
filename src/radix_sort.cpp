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

// PROBLEM STATEMENT: Radix Sort (Histogram + Prefix Sum + Scatter)
// - Radix sort never compare elements, but count each digit
// - Reuse histogram and prefix sum pattern

// INPUT: 1e7 random numbers between [0, 1e9]
// OUTPUT: sort ascending

// PASS VISUALIZATION
// input:            [170, 45, 75, 90, 802, 24, 2, 66]
// pass 1 (ones):    [170, 90, 802, 2, 24, 45, 75, 66]      ← sorted by last digit
// pass 2 (tens):    [802, 2, 24, 45, 66, 170, 75, 90]      ← sorted by tens, ties keep pass-1 order
// pass 3 (hundreds):[2, 24, 45, 66, 75, 90, 170, 802]      ← fully sorted

// OFFSET AND SCATTER VISUALIZATION
// digits:  [2, 0, 2, 1, 0, 2]
// count:   [2, 1, 3]            bin 0 has 2, bin 1 has 1, bin 2 has 3
// offset:  [0, 2, 3]            bin 0 → slots 0–1, bin 1 → slot 2, bin 2 → slots 3–5
// scatter: 1st elem (2) → slot 3, 2nd (0) → slot 0, 3rd (2) → slot 4, ...
// output:  [0, 0, 1, 2, 2, 2]

void radix_serial(vector<int>& a) {
    // In decimal digits (base 10), we have each digits 0-9 equal to 10 bins
    // - example: 1,000,000 = 1e6 + 0e5 + ... + 0e0
    // - respective digits: 1, 0, 0, 0, 0, 0, 0

    // If we use base 256, each digit can take value 0-255 equal to 256 bins
    // - example: 1,000,000 = 0*(256^3) + 15*(256^2) + 66*(256^1) + 64*(256^0)
    // - respective digits: 0, 15, 66, 64 (only need 4)

    // Why chose base 256: only need 4 digits, cheap operator, histogram 256(8-bit) = 2kB inside L1
    long N = a.size();
    vector<int> buf(N);
    int* in = a.data(); int* out = buf.data();

    // Only need 4 digits to represent
    for (int k = 0; k < 4; k++) {
        // x = i-th input 
        // x >> shift same as divide by 256^k (really means shifting the binary array)
        // & 0xFF take the last digit (8-bit)
        int shift = 8 * k;

        // Find the digit at position k, and increase counter (count = histogram 256 bins)
        long count[256] = {0};
        for (long i = 0; i < N; i++) count[(in[i] >> shift) & 0xFF]++;

        // Accumulate prefix sum based on count histogram to find start index of each digit
        // This already ensure increasing order of digit (since bin-0 smallest, bin-255 largest)
        long start[256], sum = 0;
        for (int b = 0; b < 256; b++) { start[b] = sum; sum += count[b]; }

        // Scatter: write input elements to new output positions
        for (long i = 0; i < N; i++) out[start[(in[i] >> shift) & 0xFF]++] = in[i];
        swap(in, out);
    }
    // 4 passes = even number of swaps → final result is back in a.data()
}

void radix_parallel(vector<int>& a) {
    long N = a.size();
    vector<int> buf(N);

    const int T = omp_get_max_threads();
    vector<array<long, 256>> cnt(T);    // cnt[t][b]: thread t's count for bin b (shared table)
    vector<array<long, 256>> off(T);    // off[t][b]

    #pragma omp parallel
    {
        // get thread index and fill histogram with 0
        int t = omp_get_thread_num();
        int* in = a.data(); int* out = buf.data();

        for (int k = 0; k < 4; k++) {   // every thread runs all 4 passes
            int shift = 8 * k;
            cnt[t].fill(0);

            // static = thread-0 takes index 0, 1, 2, 3; thread-1 takes index 4, 5, 6, 7 etc
            // fill in the 2D table cnt[t][b]
            // static strategy since need to preserve order (contiguous block for each thread)
            // critical assumption: within a bin, thread-0 values come before thread-1 values
            #pragma omp for schedule(static)
            for (long i = 0; i < N; i++) {
                cnt[t][(in[i] >> shift) & 0xFF]++;
            }

            // offset calculation
            //           b0 b1 b2
            // count T0:  2  1  1  -> visiting order: (b0,T0) (b0,T1) (b1,T0) (b1,T1) (b2,T0) (b2,T1)
            // count T1:  1  2  0     running sum:      0       2       3       4       6       7
            // off T0:    0  3  6        
            // off T1:    2  4  7        
            #pragma omp single
            {
                long sum = 0;
                for (int b = 0; b < 256; b++) {         // bins in order
                    for (int t2 = 0; t2 < T; t2++) {    // within a bin, threads in order
                        off[t2][b] = sum;               // where thread t2 starts writing bin b
                        sum += cnt[t2][b];
                    }
                }
            }

            // scatter own thread chunk, using only my own row of offsets
            // need same static schedule as cnt[t][b] accumulation
            // cnt[t][b] is doing the seat reserve task
            // so writing to out need to follow the reserve 
            #pragma omp for schedule(static)          // SAME static split as the counting loop
            for (long i = 0; i < N; i++) {
                int b = (in[i] >> shift) & 0xFF;
                out[off[t][b]++] = in[i];
            }
            swap(in, out);
        }
    }
}


int main() {
    // Input: N random ints, fixed seed so every version sorts the same data
    const long N = 10'000'000;
    vector<int> original(N);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, 1'000'000'000);
    for (long i = 0; i < N; i++) {
        original[i] = dist(rng);
    }

    // Reference answer for the correctness check
    vector<int> ref = original;
    sort(ref.begin(), ref.end());

    // Working copy: reset from original before EVERY sort
    vector<int> a(N);

    for (int i=0; i<2; i++) {
        cout << "Round " << i << endl;
        // Example: one timed run
        a = original;                        // copy unsorted data (outside the timer)
        {
            ScopedTimer t{"radix_serial"};
            radix_serial(a);   // inclusive range [0, N-1]
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // Example: one timed run
        a = original;                        // copy unsorted data (outside the timer)
        {
            ScopedTimer t{"radix_parallel"};
            radix_parallel(a);   // inclusive range [0, N-1]
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;
    }
    return 0;
}