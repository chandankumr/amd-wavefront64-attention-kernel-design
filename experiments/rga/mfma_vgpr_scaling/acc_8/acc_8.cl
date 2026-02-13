#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void acc_8(__global float* out)
{
    uint tid = get_local_id(0);

    half4 a = (half4)(1.0h);
    half4 b = (half4)(1.0h);

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    float4 acc4 = (float4)(0.0f);
    float4 acc5 = (float4)(0.0f);
    float4 acc6 = (float4)(0.0f);
    float4 acc7 = (float4)(0.0f);

    for (int i = 0; i < 16; ++i)
    {
        acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc0,0,0,0);
        acc1 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc1,0,0,0);
        acc2 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc2,0,0,0);
        acc3 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc3,0,0,0);
        acc4 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc4,0,0,0);
        acc5 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc5,0,0,0);
        acc6 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc6,0,0,0);
        acc7 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc7,0,0,0);
    }

    out[tid] = acc0.x + acc7.x;
}
