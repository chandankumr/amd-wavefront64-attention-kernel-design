## Occupancy Context (CDNA4 - gfx950)

VGPRs per SIMD: 256  
Allocation granularity: 8  

If a kernel uses 8 VGPR:

256 / 8 = 32 theoretical waves per SIMD

This is extremely high occupancy, so no collapse is expected.

To reduce to 2 waves per SIMD:

256 / 128 = 2 waves

This would require ~128 VGPR per thread.

The current experiments remain far below this threshold.

## Forced Accumulator Residency (acc_16_forced)

Resource usage:

- VGPR allocated: 84
- SGPR allocated: 40
- AGPR usage: up to a0–a63 (64 accumulators)

Theoretical waves per SIMD:

256 VGPR total / 88 (rounded allocation) ≈ 2 waves

This demonstrates:

- Persistent MFMA accumulators directly increase VGPR pressure
- Long accumulator lifetime reduces theoretical occupancy
- AGPR residency indirectly amplifies VGPR staging pressure

This mirrors the structural pressure pattern observed in FlashAttention-style kernels.
