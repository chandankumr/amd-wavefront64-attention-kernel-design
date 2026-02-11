# MFMA Accumulator Lifetime Experiment (CDNA4 - gfx950)

---

## Objective

Compare two MFMA usage patterns:

- **Long-lived accumulator** (persists across loop)
- **Short-lived accumulator** (writeback per iteration)

To measure impact on:

- VGPR allocation
- Live register count
- AccVGPR lifetime
- ISA scheduling behavior

This isolates accumulator lifetime effects independent of real attention complexity.

---

## Target Environment

- Architecture: **CDNA4**
- ASIC: **gfx950**
- GPU family: MI350X / MI355X
- Tool: Radeon GPU Analyzer (RGA)
- Mode: OpenCL
- Wave size: 64

---

# Long-Lived Accumulator Results

## Resource Usage (RGA)

- VGPRs allocated by HW: **8**
- Maximum live VGPRs: **4**
- SGPRs: 40
- LDS: 0
- Scratch: 0

VGPR allocation granularity: 8  
Total VGPRs per SIMD: 256  

---

## ISA Observations

Observed pattern:

v_mfma
v_mfma
v_mfma
v_mfma
v_mfma
v_mfma
v_mfma
v_mfma
s_nop
v_accvgpr_read


### Behavior

- MFMA instructions are grouped
- AccVGPR `a[0:3]` remain live across entire loop
- `v_accvgpr_read` occurs only after loop
- Accumulator lifetime spans all MFMA operations

This mimics FlashAttention-style persistent accumulation.

---

# Short-Lived Accumulator Results

## Resource Usage (RGA)

- VGPRs allocated by HW: **8**
- Maximum live VGPRs: **4**
- SGPRs: 40
- LDS: 0
- Scratch: 0

---

## ISA Observations

Observed pattern:

v_mfma
v_accvgpr_read
v_add
v_add
...
(repeated per iteration)


### Behavior

- MFMA followed immediately by accumulator readback
- AccVGPR lifetime is minimal
- Domain crossing interleaved with compute
- No persistent accumulation across loop

---

# MFMA Accumulator Lifetime Comparison (gfx950)

| Metric | Long-Lived | Short-Lived |
|--------|------------|-------------|
| Max live VGPR | 4 | 4 |
| VGPR allocated (HW) | 8 | 8 |
| MFMA grouping | 8 consecutive MFMA ops | MFMA per iteration |
| AccVGPR lifetime | Entire loop | Per iteration |
| Domain crossing | After loop | Inside loop |

---

## Key ISA Difference

### Long-Lived Version

- MFMA instructions grouped together
- AccVGPR accumulators persist across all iterations
- AccVGPR → ArchVGPR transfer happens once
- Live range spans entire loop

### Short-Lived Version

- MFMA followed immediately by readback
- AccVGPR lifetime is minimal
- Domain crossing interleaved with compute
- Live range resets every iteration

---

## Architectural Insight

This microbenchmark does **not yet show occupancy collapse** because:

- VGPR usage is very small
- Accumulator footprint is minimal
- Tile size is tiny

However, the lifetime structure is fundamentally different.

When scaled to realistic attention tile sizes:

- Long-lived accumulators amplify AccVGPR pressure
- Live ranges extend across many MFMA groups
- Register pressure compounds across wavefront-64 lanes
- Occupancy can collapse to 1 wave per SIMD

This structural lifetime amplification is the reason
FlashAttention becomes hostile to wavefront-64 at scale.

---

## Key Takeaway

The issue is not raw VGPR count in this toy example.

The issue is **lifetime structure**.

Wavefront-64 hardware amplifies long-lived state.

FlashAttention's persistent accumulators create
structural pressure that grows with tile size.

This experiment demonstrates the mechanism behind that behavior.

---

## Next Steps

- Scale accumulator footprint to force visible VGPR explosion
- Quantify waves per SIMD under increasing register pressure
- Compare with wavefront64-native accumulator design
