# Limitations and Validation Scope

## 1. Hardware Access Disclaimer

This research was conducted without direct access to CDNA4 (gfx950 / MI350-class) hardware.

All architectural conclusions are derived from:

- Radeon GPU Analyzer (RGA) compiler output
- ISA disassembly
- Live register analysis
- Resource usage reports
- Public AMD CDNA architectural documentation

No runtime benchmarking was performed on actual MI300/MI350 hardware.

---

## 2. What Is Validated

The following aspects are compiler-verified and architecturally grounded:

- VGPR allocation counts
- SGPR allocation counts
- Allocation granularity (8 VGPR blocks)
- AccVGPR → ArchVGPR transfer behavior
- MFMA grouping patterns
- Spill thresholds
- Theoretical occupancy limits (256 VGPR per SIMD)

These are not speculative; they are derived directly from the AMD compiler toolchain.

---

## 3. What Is Modeled (Not Directly Measured)

The following aspects are analytically modeled:

- Occupancy collapse behavior at HEAD_DIM ≥ 128
- Sustained MFMA throughput under low-wave occupancy
- Performance projection under register pressure constraints

These are derived from:

Waves per SIMD = floor(256 / VGPR_per_wave)

This is a hardware constraint, not a heuristic.

---

## 4. Why Compiler-Based Validation Is Legitimate

Register allocation and occupancy limits are determined at compile time.

If:

- VGPR allocation exceeds 128
- Allocation granularity rounds upward
- Waves per SIMD drop to 1

Then occupancy collapse is guaranteed independent of runtime environment.

Thus, the structural mismatch analysis between:

- FlashAttention (Warp-32 optimized)
- Wavefront-64 CDNA architecture

is valid at the architectural level.

---

## 5. Scope of Claims

This repository does NOT claim:

- Exact TFLOP measurements on MI350
- Precise runtime speedups

It DOES claim:

- A structural register pressure mismatch
- A provable occupancy collapse threshold
- A mathematically grounded wavefront-64 alternative design

---

## 6. Future Validation Path

With access to CDNA4 hardware, the following would complete the study:

- rocprof MFMA utilization metrics
- SQ_WAVES occupancy measurements
- GRBM_GUI_ACTIVE correlation
- Sustained TFLOP measurement under HEAD_DIM sweep

This repository establishes the architectural foundation for that validation.

---

## 7. Research Positioning

This work should be interpreted as:

Architecture-level kernel design analysis  
NOT hardware performance benchmarking.

The goal is structural understanding, not marketing-level performance claims.