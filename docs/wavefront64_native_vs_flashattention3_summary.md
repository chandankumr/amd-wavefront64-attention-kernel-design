# Wavefront-64 Native Attention vs FlashAttention-3  
## A Structural Comparison Under the CDNA4 Register Model

---

## 1. Executive Summary

This document summarizes the structural differences between:

- **FlashAttention-3 style persistent-tile design**
- **Wavefront-64 native split-head design**

under the CDNA4 (gfx950) register and occupancy model.

The key result:

> On Wavefront-64 GPUs, full-head persistent accumulation causes deterministic occupancy collapse when HEAD_DIM ≥ 96–128.

The Wavefront-64 native design restores occupancy by structurally partitioning accumulator ownership across lanes.

---

## 2. Hardware Model Assumptions (CDNA4 / gfx950)

| Resource | Value |
|-----------|--------|
| Wavefront size | 64 |
| VGPR per SIMD | 256 |
| SGPR per SIMD | 102 |
| MFMA execution | Separate matrix core |
| Full MFMA efficiency requires | ≥ 2 waves resident |
| Occupancy collapse threshold | ~128 VGPR per lane |

Occupancy formula:

waves_per_simd = floor(256 / used_vgpr_per_lane)


---

## 3. FlashAttention-3 Structural Model

FlashAttention-3 is architecturally improved vs FA-1/2:

- Better pipelining
- Reduced synchronization
- Improved memory overlap
- FP8 support
- Higher arithmetic intensity

However:

It preserves the fundamental tile structure:

One lane owns full HEAD_DIM accumulators


Example (HEAD_DIM = 128):

float acc[128];


Estimated VGPR footprint:

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
- No cross-wave latency hiding
- MFMA pipe under-utilized

This is a structural issue — not a compiler artifact.

---

## 4. Wavefront-64 Native Structural Model

Instead of:

1 lane → full HEAD_DIM


We design:

2 lanes → shared HEAD_DIM


Partition factor = 2  
HEAD_DIM = 128  
Per-lane accumulators = 64

float acc[64];


Estimated VGPR footprint:

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
- Latency hidden by scheduler
- MFMA issue density restored

---

## 5. Occupancy Behavior Comparison

| HEAD_DIM | FlashAttention-3 | Native Split | Advantage |
|------------|------------------|---------------|------------|
| 64 | 2 waves | 2 waves | Neutral |
| 96 | 1 wave | 2 waves | Moderate |
| 128 | 1 wave | 2 waves | Strong |
| 256 | 1 wave | 2 waves (×4 split) | Extreme |

FlashAttention-3 does not collapse on NVIDIA (warp-32) because:

- Warp size = 32
- Effective per-lane accumulator pressure is halved relative to wave-64

On CDNA4 wave-64:

- Register pressure doubles per wave
- Collapse occurs earlier

This is an architectural mismatch.

---

## 6. Performance Projection

MFMA utilization estimates:

| Scenario | Waves | Utilization |
|-----------|--------|-------------|
| Collapsed (1 wave) | 1 | 55–65% |
| Native (2 waves) | 2 | 85–95% |

Relative compute throughput:

0.90 / 0.60 ≈ 1.5×


Projected total gain (compute + memory overlap):

1.5× – 1.8×


This gain arises from occupancy recovery alone.

---

## 7. When FlashAttention-3 Remains Effective

FlashAttention-3 remains competitive when:

- HEAD_DIM ≤ 64
- Kernel is memory-bound
- FP8 reduces accumulator footprint
- Compiler shortens accumulator lifetime
- Multiple smaller heads are processed per wave

In these cases, occupancy collapse may not occur.

---

## 8. Why This Is Not an Anti-FlashAttention Claim

This analysis does not argue that FlashAttention-3 is flawed.

It argues:

> FlashAttention-3 is optimized for warp-32 hardware economics.

On warp-32:

- Accumulator ownership per lane is lower
- Register pressure scales differently
- Occupancy collapse threshold is higher

On wave-64:

- Persistent full-head ownership is structurally expensive

Therefore:

The mismatch is architectural, not algorithmic.

---

## 9. Design Philosophy Difference

FlashAttention-3 philosophy:

- Persistent accumulator
- Deep pipelining
- Maximize arithmetic intensity
- Hide latency via software scheduling

Wavefront-64 native philosophy:

- Minimize per-lane register footprint
- Preserve ≥ 2 waves
- Let hardware scheduler hide latency
- Structural register partitioning

Both are valid.

They optimize for different hardware contracts.

---

## 10. Final Architectural Conclusion

On CDNA4 Wavefront-64 GPUs:

Persistent full-head accumulators
→ deterministic occupancy collapse  
→ reduced MFMA utilization  
→ exposed latency

Wavefront-64 native split-head design:

→ restores dual-wave residency  
→ improves scheduling flexibility  
→ increases MFMA efficiency  
→ achieves closer-to-peak compute

The expected advantage grows with HEAD_DIM.

This is not a micro-optimization.

It is a structural alignment between:

Algorithm  
and  
Register-file economics.

---

## 11. Research Implication

For attention kernels targeting AMD CDNA4:

A warp-32-centric design cannot be assumed optimal.

A wavefront-64-aware accumulator partitioning strategy
should be considered a first-class design parameter.

This reframes attention kernel design as:

> A register topology problem  
not merely a tiling problem.

---