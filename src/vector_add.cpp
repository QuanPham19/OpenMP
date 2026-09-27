#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>


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

// Define serial add in vector (with copy)
void serial_add_copy(vector<double> a, vector<double> b, vector<double>& c, int size) {
    for (auto i=0; i<size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Define serial add in vector (with reference)
void serial_add_reference(vector<double>& a, vector<double>& b, vector<double>& c, int size) {
    for (auto i=0; i<size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Define parallel add (with reference)
void parallel_add_reference(vector<double>& a, vector<double>& b, vector<double>& c, int size) {
    #pragma omp parallel for schedule(static) num_threads(4)
    for (auto i=0; i<size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // omp_set_num_threads(2);
    std::printf("threads available: %d\n", omp_get_max_threads());

    for (int size: {1e4, 1e6, 1e8}) {
        cout << "vector size: " << size << endl;
        vector<double> a(size, 1.0);
        vector<double> b(size, 2.0);
        vector<double> c(size, 0.0);

        for (auto i=0; i<3; i++) {
            cout << "round: " << i << endl;
            {
                ScopedTimer t{"serial_copy"};
                serial_add_copy(a, b, c, size);
            }

            {
                ScopedTimer t{"serial_reference"};
                serial_add_reference(a, b, c, size);
            }

            {
                ScopedTimer t{"parallel_reference"};
                parallel_add_reference(a, b, c, size);
            }
        }
    }
    


}