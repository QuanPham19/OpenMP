# OpenMP
Practice and benchmark OpenMP in C++

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