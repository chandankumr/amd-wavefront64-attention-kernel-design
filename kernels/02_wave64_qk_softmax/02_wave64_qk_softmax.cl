#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// ============================================================
// Wave64 FlashAttention Stage 2
// QK + Register-Only Softmax
// Head_dim = 64
// One wave computes:
//    1 Q row  × 16 K columns per iteration
// ============================================================

__kernel void wave64_qk_softmax_stage2(
    __global const half* Q,      // [num_rows, 64]
    __global const half* K,      // [num_cols, 64]
    __global float* Out,         // [num_rows, num_cols]
    int num_cols)
{
    const int lane = get_local_id(0);     // 0..63
    const int row  = get_group_id(0);     // 1 wave per row

    // ------------------------------------------------------------
    // Persistent Q fragment
    // Each lane loads 4 FP16 values
    // ------------------------------------------------------------

    half4 q_frag = vload4(0, Q + row * 64 + lane * 4);

    // ------------------------------------------------------------
    // Loop over K tiles (16 columns per iteration)
    // ------------------------------------------------------------

    for (int col_base = 0; col_base < num_cols; col_base += 16)
    {
        float4 acc0 = (float4)(0.0f);
        float4 acc1 = (float4)(0.0f);
        float4 acc2 = (float4)(0.0f);
        float4 acc3 = (float4)(0.0f);

        // --------------------------------------------------------
        // K depth loop (64 → 4 MFMA steps)
        // --------------------------------------------------------

        for (int k = 0; k < 64; k += 16)
        {
            half4 k0 = vload4(0, K + (col_base + 0)  * 64 + k + lane * 4);
            half4 k1 = vload4(0, K + (col_base + 4)  * 64 + k + lane * 4);
            half4 k2 = vload4(0, K + (col_base + 8)  * 64 + k + lane * 4);
            half4 k3 = vload4(0, K + (col_base + 12) * 64 + k + lane * 4);

            acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(q_frag, k0, acc0, 0,0,0);
            acc1 = __builtin_amdgcn_mfma_f32_16x16x16f16(q_frag, k1, acc1, 0,0,0);
            acc2 = __builtin_amdgcn_mfma_f32_16x16x16f16(q_frag, k2, acc2, 0,0,0);
            acc3 = __builtin_amdgcn_mfma_f32_16x16x16f16(q_frag, k3, acc3, 0,0,0);
        }

        // ========================================================
        // Softmax (register-only)
        // ========================================================

        float s0  = acc0.x;
        float s1  = acc0.y;
        float s2  = acc0.z;
        float s3  = acc0.w;

        float s4  = acc1.x;
        float s5  = acc1.y;
        float s6  = acc1.z;
        float s7  = acc1.w;

        float s8  = acc2.x;
        float s9  = acc2.y;
        float s10 = acc2.z;
        float s11 = acc2.w;

        float s12 = acc3.x;
        float s13 = acc3.y;
        float s14 = acc3.z;
        float s15 = acc3.w;

        // ---- local max ----

        float local_max =
            fmax(fmax(fmax(s0, s1), fmax(s2, s3)),
                 fmax(fmax(s4, s5), fmax(s6, s7)));

        local_max = fmax(local_max,
            fmax(fmax(s8, s9), fmax(s10, s11)));

        local_max = fmax(local_max,
            fmax(fmax(s12, s13), fmax(s14, s15)));

        float row_max = sub_group_reduce_max(local_max);

        // ---- subtract + exp ----

        s0  = exp(s0  - row_max);
        s1  = exp(s1  - row_max);
        s2  = exp(s2  - row_max);
        s3  = exp(s3  - row_max);

        s4  = exp(s4  - row_max);
        s5  = exp(s5  - row_max);
        s6  = exp(s6  - row_max);
        s7  = exp(s7  - row_max);

        s8  = exp(s8  - row_max);
        s9  = exp(s9  - row_max);
        s10 = exp(s10 - row_max);
        s11 = exp(s11 - row_max);

        s12 = exp(s12 - row_max);
        s13 = exp(s13 - row_max);
        s14 = exp(s14 - row_max);
        s15 = exp(s15 - row_max);

        float local_sum =
            s0 + s1 + s2 + s3 +
            s4 + s5 + s6 + s7 +
            s8 + s9 + s10 + s11 +
            s12 + s13 + s14 + s15;

        float row_sum = sub_group_reduce_add(local_sum);
        float inv_sum = 1.0f / row_sum;

        s0  *= inv_sum;  s1  *= inv_sum;
        s2  *= inv_sum;  s3  *= inv_sum;
        s4  *= inv_sum;  s5  *= inv_sum;
        s6  *= inv_sum;  s7  *= inv_sum;
        s8  *= inv_sum;  s9  *= inv_sum;
        s10 *= inv_sum;  s11 *= inv_sum;
        s12 *= inv_sum;  s13 *= inv_sum;
        s14 *= inv_sum;  s15 *= inv_sum;

        // ========================================================
        // Correct Store Pattern (NO dynamic indexing)
        // ========================================================

        int out_base = row * num_cols + col_base;

        // Only first 16 lanes participate
        if (lane < 16)
        {
            float value;

            // Each lane selects its owned element
            switch (lane)
            {
                case 0:  value = s0;  break;
                case 1:  value = s1;  break;
                case 2:  value = s2;  break;
                case 3:  value = s3;  break;
                case 4:  value = s4;  break;
                case 5:  value = s5;  break;
                case 6:  value = s6;  break;
                case 7:  value = s7;  break;
                case 8:  value = s8;  break;
                case 9:  value = s9;  break;
                case 10: value = s10; break;
                case 11: value = s11; break;
                case 12: value = s12; break;
                case 13: value = s13; break;
                case 14: value = s14; break;
                default: value = s15; break;
            }

            Out[out_base + lane] = value;
        }
    }
}