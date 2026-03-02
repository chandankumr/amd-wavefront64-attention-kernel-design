#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void high_pressure_persistent(__global float* out,
                                       __global float* in)
{
    int gid = get_global_id(0);

    // Proper CDNA4 operands
    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    // 16 independent persistent accumulators (AGPR-backed)
    float4 acc0  = (float4)(0.0f);
    float4 acc1  = (float4)(0.0f);
    float4 acc2  = (float4)(0.0f);
    float4 acc3  = (float4)(0.0f);
    float4 acc4  = (float4)(0.0f);
    float4 acc5  = (float4)(0.0f);
    float4 acc6  = (float4)(0.0f);
    float4 acc7  = (float4)(0.0f);
    float4 acc8  = (float4)(0.0f);
    float4 acc9  = (float4)(0.0f);
    float4 acc10 = (float4)(0.0f);
    float4 acc11 = (float4)(0.0f);
    float4 acc12 = (float4)(0.0f);
    float4 acc13 = (float4)(0.0f);
    float4 acc14 = (float4)(0.0f);
    float4 acc15 = (float4)(0.0f);

    // Extra temporaries to increase VGPR lifetime
    float tmp0, tmp1, tmp2, tmp3;
    float tmp4, tmp5, tmp6, tmp7;

    for (int i = 0; i < 32; ++i)
    {
        // Global loads (force memory pressure)
        tmp0 = in[gid + i];
        tmp1 = in[gid + i + 64];
        tmp2 = in[gid + i + 128];
        tmp3 = in[gid + i + 192];

        // MFMA accumulation (persistent)
        acc0  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc0,  0,0,0);
        acc1  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc1,  0,0,0);
        acc2  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc2,  0,0,0);
        acc3  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc3,  0,0,0);
        acc4  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc4,  0,0,0);
        acc5  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc5,  0,0,0);
        acc6  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc6,  0,0,0);
        acc7  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc7,  0,0,0);
        acc8  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc8,  0,0,0);
        acc9  = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc9,  0,0,0);
        acc10 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc10, 0,0,0);
        acc11 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc11, 0,0,0);
        acc12 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc12, 0,0,0);
        acc13 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc13, 0,0,0);
        acc14 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc14, 0,0,0);
        acc15 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc15, 0,0,0);

        // Delay usage of loaded values
        tmp4 = tmp0 + tmp1;
        tmp5 = tmp2 + tmp3;
        tmp6 = tmp4 * 0.5f;
        tmp7 = tmp5 * 0.25f;
    }

    // Reduction only AFTER loop (forces persistence)
    float sum =
        acc0.x  + acc1.x  + acc2.x  + acc3.x  +
        acc4.x  + acc5.x  + acc6.x  + acc7.x  +
        acc8.x  + acc9.x  + acc10.x + acc11.x +
        acc12.x + acc13.x + acc14.x + acc15.x +
        tmp4 + tmp5 + tmp6 + tmp7;

    out[gid] = sum;
}
