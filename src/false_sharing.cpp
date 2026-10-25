#include <omp.h>
#include <iostream>
#include <chrono>
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

const int MAXT = 64;

// One counter per cache line (64 bytes each)
struct alignas(64) Padded { long v; };

// Packed: all counters next to each other -> share one cache line
alignas(64) long cnt_packed[MAXT];
// Padded: each counter on its own cache line
Padded cnt_padded[MAXT];
// Local: final result written once per thread
alignas(64) long cnt_local[MAXT];
// Serial: a single counter
alignas(64) long cnt_serial;

// Reference: one thread doing ITER increments.
// Ideal parallel versions take the same time (each thread also does ITER).
void count_serial(long ITER) {
    volatile long* c = &cnt_serial;       // volatile: force a real memory write every time
    for (long i = 0; i < ITER; i++) *c += 1;
}

// False sharing: thread t writes cnt_packed[t], all on the same cache line
void count_packed(long ITER) {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        volatile long* c = &cnt_packed[t];
        for (long i = 0; i < ITER; i++) *c += 1;
    }
}

// Padding: thread t writes cnt_padded[t].v, each on its own cache line
void count_padded(long ITER) {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        volatile long* c = &cnt_padded[t].v;
        for (long i = 0; i < ITER; i++) *c += 1;
    }
}

// Local variable: count privately, write the shared array once at the end
void count_local(long ITER) {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        volatile long local = 0;          // lives on this thread's own stack
        for (long i = 0; i < ITER; i++) local += 1;
        cnt_local[t] = local;
    }
}

// Every thread's counter must equal ITER
bool check(const long* c, int stride_longs, int T, long ITER) {
    for (int t = 0; t < T; t++)
        if (c[t * stride_longs] != ITER) return false;
    return true;
}

void reset_all() {
    for (int t = 0; t < MAXT; t++) {
        cnt_packed[t] = 0;
        cnt_padded[t].v = 0;
        cnt_local[t] = 0;
    }
    cnt_serial = 0;
}

int main() {
    const long ITER = 100'000'000;        // increments per thread
    const int T = omp_get_max_threads();
    if (T > MAXT) { cout << "Too many threads\n"; return 1; }

    cout << "T=" << T << " ITER=" << ITER << " (per thread)\n";
    cout << "ns/increment = time_ms * 1e6 / ITER\n";

    const int stride_padded = sizeof(Padded) / sizeof(long);   // 8 longs per Padded

    for (int r = 0; r < 2; r++) {
        cout << "round: " << r << endl;
        reset_all();

        {
            ScopedTimer t{"count_serial (1 thread, ITER increments)"};
            count_serial(ITER);
        }
        cout << "  correct: " << (cnt_serial == ITER ? "yes" : "NO") << endl;

        {
            ScopedTimer t{"count_packed"};
            count_packed(ITER);
        }
        cout << "  correct: " << (check(cnt_packed, 1, T, ITER) ? "yes" : "NO") << endl;

        {
            ScopedTimer t{"count_padded"};
            count_padded(ITER);
        }
        cout << "  correct: " << (check(&cnt_padded[0].v, stride_padded, T, ITER) ? "yes" : "NO") << endl;

        {
            ScopedTimer t{"count_local"};
            count_local(ITER);
        }
        cout << "  correct: " << (check(cnt_local, 1, T, ITER) ? "yes" : "NO") << endl;
    }

    return 0;
}