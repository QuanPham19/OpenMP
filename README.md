# OpenMP
Practice and benchmark OpenMP in C++

# 27 Sep 2026 (Matrix Multiplication)

**FLOP**
- Floating point operation: one add/multiply on a double
- GFLOP = 1e9 operations

**Problem statement**
- Flat array A (MxN) and array B (NxP)
- Output array C (MxP)
- Number of operations = 2(M)(N)(P) = 12 GFLOP
- Standard settings: (2000, 3000, 1000)

**Version 0: matrix_multiply_naive**
- For each (m, p) coordinate in C, loop over n to find all A(m, n) and B(n, p) and sum product
- Speed: 15ms (very slow)
    - A only slides by 1 position in memory: $A[m*N + n]$
    - B slides by 1000 positions (P) in memory: $B[n*P + p]$
    - B will jump $8KB$ for each read (different cache line, different page)
- Change order of loop (m, p) to (p, m) reduce to 14.7ms (7ms if parallel) since accessing flat index of C is better and continuous (better but still very slow)

**Version 1: matrix_multiply_advanced**
- Effect: 0.7ms on parallel (10x faster than the previous)
- Example:
```
A = | 1 2 3 |        B = | row0 |   (each row of B is P numbers)
    | 4 5 6 |            | row1 |
                         | row2 |
```

- Previous version: fill C one number at a time
```
C[0][0] = 1·B[0][0] + 2·B[1][0] + 3·B[2][0]     ← walks down column 0 of B
C[0][1] = 1·B[0][1] + 2·B[1][1] + 3·B[2][1]     ← walks down column 1 of B
```

- This version: fill the whole row C at once
    - For each row of C, start with all 0s
    - For each row of B, collect scale factor $a$ from A. Then fill the whole row of C with (a x Row of B)
    - Idea change: 
        - Before: require column jump on B
        - After: follow the flat rows (no column jump) for B and C and enhance L1 usage
```
C row 0 = 1 × row0
        + 2 × row1
        + 3 × row2
```

**Version 2: matrix_multiply_ultimate**
- Effect: 0.5ms on parallel (1.2-1.4x faster than previous)
- Previous version: 
    - Update 1 row of C at a time 
    - Need to refetch row of B several times
- This version:
    - Update 4 rows of C at a time (by storing 4 scale factor from A)
    - Need to refetch row of B less frequently
    - 4 rows in C to be updated together fit in L1 cache (32KB)
- Moral of the story:
    - In hot loops, make data load as convenient (contiguous, predictable) as possible
    - Re-use data that is still fresh on cache
    - Design the loop to fit into L1/L2 cache where it is super fast

# 27 Sep 2026 (Vector Addition)

**CPU**
- AMD EPYC 7763 (Zen 3, "Milan"), 64-core server chip, shared with other VMs
- VM gets 4 vCPUs = 2 physical cores × 2 hyperthreads (SMT)
- Per core: 2 FP add units, AVX2 (4 doubles per instruction), ~24 billion double adds/s peak

**Cache and Memory**
- L1d: 32 KB per core (64 KB total, 2 instances)
- L1i: 32 KB per core (64 KB total, 2 instances)
- L2: 512 KB per core (1 MB total, 2 instances)
- L3: 32 MB, shared by both cores (and possibly other tenants on the same chiplet)
- RAM: 15 GiB total, ~11 GiB available

**Measured bandwidth (vector add, 32 B/element, best of 3)**
- L2-resident (1e4 elements, 240 KB): ~93 GB/s on 1 thread
- Mostly L3-resident (1e6, 24 MB): ~42 GB/s serial → ~68 GB/s with 4 threads
- DRAM (1e8, 2.4 GB): ~37 GB/s serial → ~48 GB/s with 4 threads (practical ceiling)

| Size | Data (a+b+c) | Where it lives  | Serial             | Parallel (nthread=4)          | Speedup |
|------|--------------|-----------------|--------------------|--------------------|---------|
| 1e4  | 240 KB       | L2 cache        | ~93 GB/s (3.5 µs)  | ~84 GB/s (3.8 µs)  | ~0.90x  |
| 1e6  | 24 MB        | mostly L3 cache | ~42 GB/s (0.76 ms) | ~68 GB/s (0.47 ms) | ~1.62x  |
| 1e8  | 2.4 GB       | DRAM            | ~37 GB/s (85 ms)   | ~48 GB/s (66 ms)   | ~1.28x  |

*Minimum of 3 rounds. Bandwidth assumes 32 bytes/element (read a, read b, write c + write-allocate of c). Run on a 4-vCPU GitHub Codespace (AMD EPYC 7763, 2 cores × 2 threads).*

**Notes**
- This exercise is cheap on computation, the difference is mostly on memory bandwidth
- Memory-bound loops gain little from threads once data is in DRAM (~1.3x)
- There are still speed-up since 1 core does not utilize all RAM bandwidth
- Small loops (< ~1e5 elements) lose to parallel overhead
- Results vary run to run due to shared hardware and vCPU scheduling