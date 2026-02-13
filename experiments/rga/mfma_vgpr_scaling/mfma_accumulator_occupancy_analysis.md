# MFMA Accumulator Occupancy Analysis  
### CDNA4 (gfx950) – AGPR Lifetime vs Wave Occupancy

---

## 1. Objective

This experiment investigates how **MFMA accumulator lifetime** impacts:

- VGPR allocation
- AccVGPR (AGPR) residency
- Wave occupancy per SIMD
- Structural pressure relevant to FlashAttention-style kernels

The goal is to determine:

> At what point does persistent accumulator residency begin to collapse occupancy on wavefront-64 hardware?

---

## 2. Experimental Setup

**Architecture:** CDNA4  
**ASIC:** gfx950  
**Tool:** Radeon GPU Analyzer (RGA)  
**Mode:** OpenCL  
**Wavefront size:** 64  
**VGPRs per SIMD:** 256  
**Allocation granularity:** 8 VGPR  

All kernels use:

__builtin_amdgcn_mfma_f32_16x16x16f16


Each variant increases the number of independent accumulators kept live across the loop.

---

## 3. Kernel Variants

| Variant | Description |
|---------|------------|
| acc_1 | Single persistent accumulator |
| acc_4 | Four independent accumulators |
| acc_8 | Eight independent accumulators |
| acc_16 | Sixteen independent accumulators |
| acc_16_forced | Sixteen accumulators with forced lifetime extension preventing compiler reuse |

The `acc_16_forced` version intentionally defeats compiler register reuse to expose true AGPR residency cost.

---

## 4. Measured Resource Usage

From RGA output:

| Kernel | VGPR Used | VGPR Allocated (HW) |
|--------|-----------|---------------------|
| acc_1 | 8 | 8 |
| acc_4 | 8 | 8 |
| acc_8 | 8 | 8 |
| acc_16 | 8 | 8 |
| acc_16_forced | 84 | 88 (granularity padded) |

---

## 5. Occupancy Analysis

Wave occupancy per SIMD is bounded by:

waves_per_simd = floor(256 / VGPR_per_wave)

### For acc_1 → acc_16

256 / 8 = 32 waves per SIMD


No occupancy pressure observed.

The compiler aggressively reuses AGPRs and collapses live ranges.

---

### For acc_16_forced

256 / 84 ≈ 3 waves

Accounting for granularity (88 allocated):

256 / 88 = 2 waves


### Occupancy Collapse Observed

| Kernel | Theoretical Waves per SIMD |
|--------|----------------------------|
| acc_1 | 32 |
| acc_4 | 32 |
| acc_8 | 32 |
| acc_16 | 32 |
| acc_16_forced | 2–3 waves |

This is a **>10× reduction in occupancy**.

---

## 6. ISA Behavior Differences

### Non-forced variants

- MFMA instructions reuse accumulator ranges
- Live ranges are short
- AGPR pressure remains bounded
- Domain crossing delayed
- Compiler performs aggressive lifetime collapsing

### Forced variant

- Large number of AGPRs initialized (`v_accvgpr_write_b32`)
- MFMA groups operate on wide accumulator ranges
- Accumulators persist across loop iterations
- Significant `v_accvgpr_read_b32` fan-in at epilogue
- VGPR pressure spikes

This reveals the real structural cost of persistent accumulator residency.

---

## 7. Architectural Insight

This experiment demonstrates:

> AGPR pressure is not visible unless accumulator lifetime is intentionally preserved.

Compiler optimizations hide pressure in small test kernels.

However, large-scale attention kernels:

- Maintain multiple accumulators
- Fuse softmax and normalization
- Keep tiles live across long loops
- Prevent early accumulator retirement

When scaled to realistic head sizes, this pattern becomes structurally hostile to:

- Wavefront-64 occupancy
- VGPR residency
- SIMD utilization

---

## 8. Implications for FlashAttention on Wave64

FlashAttention-style kernels:

- Keep accumulators live across many MFMA groups
- Perform delayed reduction
- Interleave memory + compute
- Maintain softmax state

On wave64 architectures:

- AccVGPR pressure extends across long loops
- Live ranges grow with tile size
- VGPR allocation reduces waves per SIMD
- Latency hiding collapses

This is not a micro-optimization issue.

It is a structural scheduling problem.

---

## 9. Conclusion

This controlled experiment shows:

1. Compiler reuse masks AGPR pressure in small cases
2. Persistent accumulator residency causes rapid VGPR growth
3. Occupancy collapses once VGPR exceeds ~80 per wave
4. Wavefront-64 designs are sensitive to accumulator lifetime scaling

The `acc_16_forced` variant provides a minimal reproducible demonstration of:

> How persistent MFMA accumulators can collapse wave occupancy on CDNA4.

This pattern directly informs attention kernel design for wavefront-64 GPUs.

---

## 10. Next Steps

Potential follow-ups:

- Scale to 32 accumulators
- Measure live register count from `livereg.txt`
- Add occupancy vs accumulator plot
- Compare against wave32 behavior
- Implement wavefront64-native attention prototype

---

This experiment transitions from microbenchmark to architectural signal:

> Accumulator lifetime is an occupancy control variable.

That is the core takeaway.
