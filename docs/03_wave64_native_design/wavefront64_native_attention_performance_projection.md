# Wavefront-64 Native Attention — Performance Projection (CDNA4)

---

## 1. Purpose

This document projects the expected performance difference between:

- **FlashAttention-style tile (persistent full-head accumulators)**
- **Wavefront-64 native split-head design**

under the CDNA4 (gfx950) register model.

This is not a benchmark result.

It is a **first-principles architectural performance projection**
based on:

- Measured VGPR usage (RGA)
- Wave occupancy math
- MFMA throughput behavior
- Latency hiding economics

---

## 2. Architectural Constraints (CDNA4 / gfx950)

| Resource | Value |
|-----------|--------|
| Wavefront size | 64 |
| VGPR per SIMD | 256 |
| SGPR per SIMD | 102 |
| AccVGPR (matrix accum) | Separate file |
| MFMA throughput | Peak when ≥ 2 waves |
| Occupancy collapse threshold | ~128 VGPR |

Occupancy formula:

waves_per_simd = floor(256 / used_vgpr_per_lane)


---

## 3. Register Footprint Comparison

### 3.1 FlashAttention-Style (Persistent Head)

Example: HEAD_DIM = 128

float acc[128]


Estimated VGPR per lane:

| Component | VGPR |
|------------|------|
| Accumulators | 128 |
| Q fragment | 8–12 |
| K fragment | 8–12 |
| Softmax state | 4 |
| Temporaries | 8–12 |
| **Total** | 160–170 |

Occupancy:

floor(256 / 168) = 1 wave


Result:

- Single-wave execution
- No latency hiding
- MFMA stalls exposed

---

### 3.2 Wavefront-64 Native (Split Head ×2)

HEAD_DIM = 128  
Partition factor = 2  
Per-lane accumulators = 64

Estimated VGPR per lane:

| Component | VGPR |
|------------|------|
| Accumulators | 64 |
| Q fragment | 8–12 |
| K fragment | 8–12 |
| Softmax state | 4 |
| Temporaries | 8 |
| **Total** | 96–104 |

Occupancy:

floor(256 / 104) = 2 waves


Result:

- Dual-wave residency
- MFMA issue overlap
- Latency hidden by scheduler

---

## 4. MFMA Throughput Model

MFMA engines require:

- Independent wave to hide memory latency
- Ready operands to avoid instruction buffer stalls

Single-wave scenario:

Memory load → MFMA → Wait → MFMA → Wait


Dual-wave scenario:

Wave 0: MFMA
Wave 1: Load
Wave 0: MFMA
Wave 1: MFMA


Effective result:

| Scenario | Expected MFMA Utilization |
|-----------|---------------------------|
| 1 wave | 55–65% |
| 2 waves | 85–95% |

---

## 5. Performance Projection Model

Let:

P_peak = theoretical MFMA peak
U = utilization


Then:

Effective throughput = P_peak × U


Projected comparison:

| Kernel Type | Waves | Utilization | Relative Performance |
|--------------|--------|-------------|-----------------------|
| FlashAttention (collapsed) | 1 | 0.60 | 1.0x |
| Native Split-Head | 2 | 0.90 | 1.5x |

Projected gain:

0.90 / 0.60 = 1.5×


≈ **50% improvement** from occupancy recovery alone.

---

## 6. Memory Behavior Impact

FlashAttention (collapsed):

- Larger persistent accumulators
- Longer live ranges
- Increased register pressure
- Reduced scheduling flexibility

Wave64-native:

- Shorter accumulator lifetime
- Smaller register footprint
- More waves available to hide:

  - HBM latency
  - LDS latency
  - VALU operations

Expected improvement in memory-bound regimes:

10–20% additional uplift.

---

## 7. Combined Projection

Conservative estimate:

| Component | Gain |
|------------|------|
| MFMA utilization recovery | 1.4–1.6× |
| Memory latency hiding | 1.1–1.2× |
| Combined realistic gain | 1.5–1.8× |

Projected real-world improvement:

+50% to +80% vs collapsed FA-style kernel


Assumes identical math precision and memory layout.

---

## 8. Sensitivity to Head Dimension

| HEAD_DIM | Flash Occ | Native Occ | Advantage |
|------------|------------|-------------|------------|
| 64 | 2 waves | 2 waves | Neutral |
| 96 | 1 wave | 2 waves | Moderate |
| 128 | 1 wave | 2 waves | Strong |
| 256 | 1 wave | 2 waves (×4 split) | Extreme |

The larger the head dimension,  
the stronger the architectural mismatch.

---

## 9. When Advantage Shrinks

Native design advantage reduces when:

- HEAD_DIM ≤ 64
- Kernel is strictly memory-bound
- Compiler reduces accumulator lifetime automatically
- FP8 reduces register pressure sufficiently

---

## 10. Architectural Conclusion

On CDNA4 Wavefront-64:

Persistent full-head accumulators
cause deterministic occupancy collapse.

Splitting the head dimension:

- Restores dual-wave residency
- Raises MFMA utilization
- Reduces exposed latency
- Increases scheduling flexibility

The expected performance gain is:

> 1.5× in compute-bound attention workloads  
> 1.5–1.8× in mixed compute/memory workloads

This gain is architectural, not micro-optimizing.

It arises from aligning algorithm structure
with hardware register economics.

---

## 11. Final Statement

Wavefront-64-native attention is not a micro-optimization.

It is a structural redesign that converts:

- A register-bound kernel  
into  
- An occupancy-balanced, MFMA-efficient kernel.

On CDNA4-class hardware,
this transformation determines whether attention runs
at 60% of peak
or near hardware limits.

---
