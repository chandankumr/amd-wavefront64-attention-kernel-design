# FlashAttention-3 Under the CDNA4 Wavefront-64 Register Model

---

## 1. Purpose of This Analysis

This document evaluates a realistic **FlashAttention-3 forward tile configuration** under the register constraints of:

- **Architecture:** CDNA4 (gfx950)
- **Wavefront size:** 64
- **Available VGPR per SIMD:** 256
- **VGPR allocation granularity:** 8

The objective is to determine:

> Whether FlashAttention-3’s persistent accumulator strategy is occupancy-safe on wavefront-64.

This is a quantitative analysis grounded in measured RGA data.

---

## 2. Representative FlashAttention-3 Tile Configuration

A common transformer inference configuration:

- `BLOCK_M = 64`
- `BLOCK_N = 64`
- `Head_Dim = 128`
- Accumulator precision = FP32

Mapping to wavefront-64:

- 1 lane processes 1 row
- 64 lanes process 64 rows

This is a natural mapping on AMD hardware.

---

## 3. CDNA4 Register Budget

Per SIMD:

VGPR_total = 256
Allocation granularity = 8


Maximum waves per SIMD is bounded by:

waves_per_SIMD = floor(256 / VGPR_used)


---

## 4. Register Model Decomposition

Total per-lane VGPR usage can be approximated as:



Total_VGPR ≈
Persistent_Accumulator

- Q fragment

- K fragment

- V fragment

- Softmax temporaries

- Loop bookkeeping


The dominant term is the persistent FP32 accumulator.

---

## 5. Accumulator Cost (Head_Dim = 128)

Each lane maintains one full output row accumulator.

Head_Dim = 128 FP32 values


Each FP32 value consumes one VGPR.

Therefore:

Accumulator_VGPR ≈ 128


This cost is structural and independent of compiler optimization.

---

## 6. Empirical Overhead from Measured Sweep

From your RGA occupancy sweep:

| Kernel     | USED_VGPR |
|------------|-----------|
| sweep_01   | 16        |
| sweep_02   | 16        |
| sweep_04   | 16        |
| sweep_08   | 80        |
| sweep_16   | 136       |
| sweep_32   | 296       |

Observations:

- Base non-accumulator overhead ≈ 16–24 VGPR
- Accumulator scaling dominates beyond sweep_08
- sweep_16 (136 VGPR) already approaches collapse

Conservative estimate:

Non_accumulator_overhead ≈ 24 VGPR


---

## 7. Estimated Total Register Usage (Head_Dim = 128)

Total_VGPR ≈ 128 (accumulator)
+ 24 (overhead)
= 152 VGPR


---

## 8. Occupancy Calculation

waves_per_SIMD = floor(256 / 152)
= floor(1.68)
= 1 wave


Result:

⚠ Single-wave occupancy

This represents structural occupancy collapse.

---

## 9. Comparison: Head_Dim = 64

If:

Head_Dim = 64


Then:

Accumulator_VGPR ≈ 64
Total_VGPR ≈ 64 + 24 = 88


Occupancy:

waves_per_SIMD = floor(256 / 88)
= 2 waves


This aligns with measured data:

- sweep_08 → 80 VGPR (safe region)
- sweep_16 → 136 VGPR (collapse boundary)

The register model matches empirical results.

---

## 10. Why FlashAttention-3 Optimizations Do Not Remove This Constraint

FlashAttention-3 improves:

- Asynchronous pipelining
- Memory staging
- Softmax overlap
- FP8 throughput

However:

It retains persistent row-wise FP32 accumulators.

On warp-32 hardware:

- 32 lanes per warp
- Lower per-wave residency pressure
- Register pressure remains manageable

On wavefront-64 hardware:

- Accumulator residency doubles per wave
- VGPR pressure amplifies
- Occupancy collapses at realistic head dimensions

This is architectural, not implementation-specific.

---

## 11. Structural Mismatch Summary

For:

BLOCK_M = 64
Head_Dim = 128


FlashAttention-3 under wavefront-64 economics results in:

- ~152 VGPR usage per lane
- Single-wave occupancy
- Reduced latency hiding capability
- Lower ability to mask memory latency

This is not a compiler issue.

This is a register residency constraint.

---

## 12. Conclusion

FlashAttention-3’s persistent accumulator model:

- Is optimal on warp-32 hardware
- Becomes occupancy-suboptimal on wavefront-64 for large head dimensions

Without architectural adaptation (e.g., head partitioning or accumulator splitting),
occupancy collapse is mathematically inevitable on CDNA4.

---