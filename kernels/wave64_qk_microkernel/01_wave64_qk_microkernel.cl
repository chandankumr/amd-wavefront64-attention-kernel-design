#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ================================================================
// Wave64 QK Microkernel (Stage 1)
// Head_dim = 64 (locked)
// One wave computes 1 Q row × 16 K rows
//
// Register Model:
//
// VGPR:
//   q_frag   → 2 VGPR (half4)
//   k_frag   → 2 VGPR (half4)
//   temp     → few VGPR
//
// ACCVGPR Layout (16 total):
//   Tile 0 → a[0:3]
//   Tile 1 → a[4:7]
//   Tile 2 → a[8:11]
//   Tile 3 → a[12:15]
//
// Each tile accumulates 4 columns.
// Total columns per wave = 16.
// ================================================================

__kernel void wave64_qk_microkernel(
    __global const half* Q,   // [num_rows, 64]
    __global const half* K,   // [num_cols, 64]
    __global float* Out,      // [num_rows, num_cols]
    int num_cols              // sequence length
)
{
    // ------------------------------------------------------------
    // Wave + row mapping
    // ------------------------------------------------------------
    const int lane = get_local_id(0);     // 0..63
    const int row  = get_group_id(0);     // one wave per Q row

    // ------------------------------------------------------------
    // Persistent Q fragment
    // Each lane owns 1 element of head_dim=64.
    // We load as half4 chunks across depth steps.
    // ------------------------------------------------------------

    // We process depth in 4 chunks of 16
    // Each chunk uses half4 load per lane
    // Q row base pointer:
    const int q_base = row * 64;

    // ------------------------------------------------------------
    // Accumulator tiles (persistent across full K sweep)
    // 4 tiles × 4 ACCVGPR each = 16 ACCVGPR
    // ------------------------------------------------------------
    float4 acc0  = (float4)(0.0f);  // a[0:3]
    float4 acc1  = (float4)(0.0f);  // a[4:7]
    float4 acc2  = (float4)(0.0f);  // a[8:11]
    float4 acc3  = (float4)(0.0f);  // a[12:15]

    // ------------------------------------------------------------
    // Outer loop over K rows in blocks of 16 columns
    // Each iteration computes 16 output columns
    // ------------------------------------------------------------
    for (int col_block = 0; col_block < num_cols; col_block += 16)
    {
        // Reset accumulators for this 16-column block
        acc0 = (float4)(0.0f);
        acc1 = (float4)(0.0f);
        acc2 = (float4)(0.0f);
        acc3 = (float4)(0.0f);

        // --------------------------------------------------------
        // Depth sweep (64 = 4 × 16)
        // Each depth step feeds MFMA
        // --------------------------------------------------------
        for (int d = 0; d < 64; d += 16)
        {
            // ---------------------------
            // Load Q fragment (persistent per row)
            // ---------------------------
            half4 q_frag = vload4(0, Q + q_base + d + lane);

            // ---------------------------
            // For each of 4 output tiles
            // ---------------------------

            // Tile 0 → columns col_block + 0..3
            half4 k0 = vload4(0,
                K + (col_block + 0) * 64 + d + lane);

            // Tile 1 → columns col_block + 4..7
            half4 k1 = vload4(0,
                K + (col_block + 4) * 64 + d + lane);

            // Tile 2 → columns col_block + 8..11
            half4 k2 = vload4(0,
                K + (col_block + 8) * 64 + d + lane);

            // Tile 3 → columns col_block + 12..15
            half4 k3 = vload4(0,
                K + (col_block + 12) * 64 + d + lane);

            // ---------------------------
            // MFMA accumulation
            // Accumulators persist across depth loop
            // ---------------------------
            acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        q_frag, k0, acc0, 0, 0, 0);

            acc1 = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        q_frag, k1, acc1, 0, 0, 0);

            acc2 = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        q_frag, k2, acc2, 0, 0, 0);

            acc3 = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        q_frag, k3, acc3, 0, 0, 0);
        }

        // --------------------------------------------------------
        // Write results
        // Each lane contributes one element per tile
        // --------------------------------------------------------
        const int out_base = row * num_cols + col_block;

        Out[out_base + 0  + lane] = acc0.x;
        Out[out_base + 4  + lane] = acc1.x;
        Out[out_base + 8  + lane] = acc2.x;
        Out[out_base + 12 + lane] = acc3.x;
    }
}