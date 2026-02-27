# Wavefront-64 Native Attention Kernel — Pseudocode Specification

---

## 1. Purpose

This document specifies a **wavefront-64-native forward attention kernel**
in structured pseudocode form.

It integrates:

- Head-dimension partitioning
- Dual-wave residency constraints
- MFMA scheduling awareness
- Online softmax handling
- Cross-lane accumulator reduction

Target architecture:

- CDNA4 (gfx950)
- Wavefront size: 64
- VGPR budget per SIMD: 256

This is not framework pseudocode.
This is ISA-aware algorithm mapping.

---

## 2. Design Parameters

BLOCK_M = 64
BLOCK_N = 64
HEAD_DIM = 128
PARTITION_FACTOR = 2
HEAD_SLICE = HEAD_DIM / PARTITION_FACTOR = 64


Each wave:

- Processes 32 rows (since 2 lanes per row)
- Uses all 64 lanes
- Maintains ≤ 104 VGPR per lane

---

## 3. Lane Mapping

lane_id = get_lane_id()

row_id = lane_id / PARTITION_FACTOR
head_slice_id = lane_id % PARTITION_FACTOR


Mapping:

lane 0 → row 0, slice 0 (0–63)
lane 1 → row 0, slice 1 (64–127)
lane 2 → row 1, slice 0
lane 3 → row 1, slice 1
...


---

## 4. Register Allocation Model

Per lane:

float acc[HEAD_SLICE] // 64 FP32 accumulators
float q_fragment[Q_TILE] // MFMA operand
float k_fragment[K_TILE]
float softmax_max
float softmax_denom


Total expected VGPR:

Accumulator: 64
Q fragment: ~8–12
K fragment: ~8–12
Softmax state: 2–4
Temporaries: ~8
Total: 96–104


Occupancy:

floor(256 / 104) = 2 waves per SIMD


---

## 5. Kernel Structure

### 5.1 Load Q Tile

q_fragment = load_Q(row_id, head_slice_id)


Q is persistent for full KV sweep.

---

### 5.2 Initialize Accumulators

for i in 0 .. HEAD_SLICE:
acc[i] = 0


Initialize softmax state:

softmax_max = -inf
softmax_denom = 0


---

### 5.3 Loop Over K/V Tiles

for kv_tile in 0 .. seq_len step BLOCK_N:


---

#### 5.3.1 Load K Tile

k_fragment = load_K(kv_tile, head_slice_id)


---

#### 5.3.2 Compute QK^T via MFMA

score_fragment = MFMA(q_fragment, k_fragment)


MFMA writes to AccVGPR domain.

Immediately move to ArchVGPR for softmax:

score = accvgpr_read(score_fragment)


---

#### 5.3.3 Online Softmax Update

Compute tile max:

tile_max = max(score)


Wave-level reduction (row-local):

row_tile_max = wave_reduce_max(tile_max, pair_of_lanes)


Update running max:

new_max = max(softmax_max, row_tile_max)


Rescale denominator:

softmax_denom *= exp(softmax_max - new_max)


Update:

softmax_max = new_max
softmax_denom += sum(exp(score - softmax_max))


---

#### 5.3.4 Apply Softmax to V

v_fragment = load_V(kv_tile, head_slice_id)


Compute:

acc += exp(score - softmax_max) * v_fragment


All operations remain within HEAD_SLICE partition.

---

## 6. Cross-Lane Final Reduction

At end of KV sweep:

Each row has two partial accumulators:

lane_even → acc_slice_0
lane_odd → acc_slice_1


Perform pairwise reduction:

if head_slice_id == 0:
partner = shuffle(acc_slice_1 from lane+1)
acc_full = concat(acc_slice_0, partner)


Only 1 cross-lane exchange required.

Cost: negligible relative to MFMA loop.

---

## 7. Write Output

store_output(row_id, acc_full)


Each row written once after reduction.

---

## 8. Scheduling Notes (ISA-Level Intent)

To maintain MFMA throughput:

- Use stationary-A or stationary-B scheduling
- Overlap VALU softmax math with next MFMA issue
- Keep K tiles streaming through registers
- Avoid long-lived ArchVGPR temporaries

Optional optimizations:

- LDS double buffering
- XOR-based bank swizzling
- FP8 MFMA with scaled accumulation

---

## 9. Why This Design Works on Wave-64

This design:

- Cuts per-lane accumulator footprint in half
- Restores dual-wave occupancy
- Preserves MFMA density
- Limits AccVGPR lifetime
- Controls VGPR granularity waste

It aligns algorithm structure with:

- 256 VGPR per SIMD
- 8-register allocation granularity
- 64-lane execution width

---

## 10. Architectural Conclusion

Wavefront-64 hardware penalizes persistent full-head accumulators.

The correct abstraction is:

> A row is a wave-level object, not a lane-level object.

Partitioning the head dimension across lanes converts:

- A register-bound design  
into  
- An occupancy-balanced design.

This is the minimal architectural transformation required
for attention to become wave-64-native.

---
