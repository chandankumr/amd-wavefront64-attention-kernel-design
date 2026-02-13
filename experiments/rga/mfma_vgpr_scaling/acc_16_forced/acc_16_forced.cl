#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void acc_16_forced(__global float* out)
{
    uint tid = get_local_id(0);

    // Proper CDNA4 FP16 operands
    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    // 16 independent accumulators (AGPR-backed)
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

    // Loop keeps ALL accumulators live
    for (int i = 0; i < 32; ++i)
    {
        acc0  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc0 ,0,0,0);
        acc1  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc1 ,0,0,0);
        acc2  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc2 ,0,0,0);
        acc3  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc3 ,0,0,0);
        acc4  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc4 ,0,0,0);
        acc5  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc5 ,0,0,0);
        acc6  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc6 ,0,0,0);
        acc7  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc7 ,0,0,0);
        acc8  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc8 ,0,0,0);
        acc9  = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc9 ,0,0,0);
        acc10 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc10,0,0,0);
        acc11 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc11,0,0,0);
        acc12 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc12,0,0,0);
        acc13 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc13,0,0,0);
        acc14 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc14,0,0,0);
        acc15 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc15,0,0,0);
    }

    // Force all accumulators to remain live until here
    float sum =
        acc0.x  + acc1.x  + acc2.x  + acc3.x  +
        acc4.x  + acc5.x  + acc6.x  + acc7.x  +
        acc8.x  + acc9.x  + acc10.x + acc11.x +
        acc12.x + acc13.x + acc14.x + acc15.x;

    out[tid] = sum;
}
