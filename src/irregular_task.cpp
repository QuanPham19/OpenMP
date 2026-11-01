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

// Hoare partition with median-of-three pivot, inclusive range [lo, hi]
// Example:
// [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] --> index
// [  , 7, 2, 9, 4, 1, 8, 3, 6, 5,  ] --> values
// i = -1 and j = 9 (starting to slide index)
// low = 0, mid = 4, high = 8 (index)

// Step 1: re-order low/mid/high
// [  , 1, 2, 9, 4, 5, 8, 3, 6, 7,  ] --> 1, 5, 7 increasing as expected

// Step 2: swap to remove violation
// Slide from left until first a[i] > 5 --> value 9 at position 2
// Slide from right until first a[j] < 5 --> value 3 at position 6
// Swap a[2] and a[6]
// [  , 1, 2, 3, 4, 5, 8, 9, 6, 7, ] --> swap value 2 and 9 as expected

// Step 3: keep sliding i and j to mid and see no more violation
// [1, 2, 3, 4, 5 | 8, 9, 6, 7] --> ready to partition to halves and recursion

long partition(int* a, long lo, long hi) {
    long mid = lo + (hi - lo) / 2;
    if (a[mid] < a[lo]) swap(a[mid], a[lo]);
    if (a[hi]  < a[lo]) swap(a[hi],  a[lo]);
    if (a[hi]  < a[mid]) swap(a[hi], a[mid]);
    int pivot = a[mid];

    long i = lo - 1, j = hi + 1;
    while (true) {
        do i++; while (a[i] < pivot);
        do j--; while (a[j] > pivot);
        if (i >= j) return j;
        swap(a[i], a[j]);
    }
}

// Version 0: Naive without OpenMP
void qs_serial(int* a, long lo, long hi) {
    if (lo >= hi) return;
    long p = partition(a, lo, hi);
    qs_serial(a, lo, p);          // left:  [lo, p]
    qs_serial(a, p + 1, hi);      // right: [p+1, hi]
}

// Version 1: 
// #pragma omp parallel    →  4 threads exist
// #pragma omp single      →  only T0 starts sorting; T1–T3 wait for work

// T0: partition [0, N)  → pushes task "sort left", task "sort right"
// T1: grabs "sort left"  → partitions it → pushes 2 more tasks
// T2: grabs "sort right" → partitions it → pushes 2 more tasks
// T3: grabs whatever is on the pile next

// Every thread: finish your task → grab another from the pile → repeat
// Region ends when the pile is empty and every task is done
// Issue: in deeper level, the sorting is very cheap, but the task overhead/scheduling is expensive (6-7x slower than naive)

void qs_task_nocutoff(int* a, long lo, long hi) {
    if (lo >= hi) return;
    long p = partition(a, lo, hi);       // split this piece (done by whoever runs this task)

    #pragma omp task                     // put "sort left" on the pile
    qs_task_nocutoff(a, lo, p);

    #pragma omp task                     // put "sort right" on the pile
    qs_task_nocutoff(a, p + 1, hi);

    #pragma omp taskwait                 // wait for my two children before returning
}

void sort_task_nocutoff(int* a, long N) {
    #pragma omp parallel                 // create the team of threads
    {
        #pragma omp single               // one thread starts the recursion
        qs_task_nocutoff(a, 0, N - 1);   // the other threads execute the tasks it creates
    }                                    // implicit barrier: also waits for all tasks
}


// Version 1: cutoff (how large the size needs to be a 'task')
// The tradeoff:
// Cutoff too small (1, 10): millions of tiny tasks, and overhead dominates. 
// Cutoff too large (1e7): only about 20 tasks. Splits are uneven, so some threads finish early with no tasks left to grab: load imbalance 
// Sweet spot: enough tasks to keep 4 threads busy (hundreds or thousands), each big enough that ~1 µs of overhead is negligible. 
void qs_task(int* a, long lo, long hi, long cutoff) {
    if (lo >= hi) return;

    long size = hi - lo + 1;              // number of elements in this piece
    if (size < cutoff) {                  // too small to be worth a task
        qs_serial(a, lo, hi);             // sort it here, serially
        return;
    }

    long p = partition(a, lo, hi);
    #pragma omp task
    qs_task(a, lo, p, cutoff);
    #pragma omp task
    qs_task(a, p + 1, hi, cutoff);
    #pragma omp taskwait
}

void sort_task(int* a, long N, long cutoff) {
    #pragma omp parallel                 // create the team of threads (once)
    {
        #pragma omp single               // one thread starts the recursion
        qs_task(a, 0, N - 1, cutoff);    // the others pick up tasks as they appear
    }                                    // barrier: waits until every task is done
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
            ScopedTimer t{"qs_serial"};
            qs_serial(a.data(), 0, N - 1);   // inclusive range [0, N-1]
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        a = original;                        // copy unsorted data (outside the timer)
        {
            ScopedTimer t{"sort_task_nocutoff"};
            sort_task_nocutoff(a.data(), N);   // inclusive range [0, N-1]
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // usage
        a = original;
        {
            ScopedTimer t{"sort_task cutoff=100"};
            sort_task(a.data(), N, 100);
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // usage
        a = original;
        {
            ScopedTimer t{"sort_task cutoff=1000"};
            sort_task(a.data(), N, 1000);
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // usage
        a = original;
        {
            ScopedTimer t{"sort_task cutoff=10000"};
            sort_task(a.data(), N, 10000);
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // usage
        a = original;
        {
            ScopedTimer t{"sort_task cutoff=100000"};
            sort_task(a.data(), N, 100000);
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

        // usage
        a = original;
        {
            ScopedTimer t{"sort_task cutoff=1000000"};
            sort_task(a.data(), N, 1000000);
        }
        cout << "correct: " << (a == ref ? "yes" : "NO") << endl;

    }

}