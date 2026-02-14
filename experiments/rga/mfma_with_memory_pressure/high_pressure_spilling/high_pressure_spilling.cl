#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void high_pressure_spilling(__global float* out,
                                     __global float* in)
{
    int gid = get_global_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    // 32 independent accumulators
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
    float4 acc16 = (float4)(0.0f);
    float4 acc17 = (float4)(0.0f);
    float4 acc18 = (float4)(0.0f);
    float4 acc19 = (float4)(0.0f);
    float4 acc20 = (float4)(0.0f);
    float4 acc21 = (float4)(0.0f);
    float4 acc22 = (float4)(0.0f);
    float4 acc23 = (float4)(0.0f);
    float4 acc24 = (float4)(0.0f);
    float4 acc25 = (float4)(0.0f);
    float4 acc26 = (float4)(0.0f);
    float4 acc27 = (float4)(0.0f);
    float4 acc28 = (float4)(0.0f);
    float4 acc29 = (float4)(0.0f);
    float4 acc30 = (float4)(0.0f);
    float4 acc31 = (float4)(0.0f);

    for (int i = 0; i < 32; ++i)
    {
        float val = in[gid + i];

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
        acc16 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc16,0,0,0);
        acc17 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc17,0,0,0);
        acc18 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc18,0,0,0);
        acc19 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc19,0,0,0);
        acc20 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc20,0,0,0);
        acc21 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc21,0,0,0);
        acc22 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc22,0,0,0);
        acc23 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc23,0,0,0);
        acc24 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc24,0,0,0);
        acc25 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc25,0,0,0);
        acc26 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc26,0,0,0);
        acc27 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc27,0,0,0);
        acc28 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc28,0,0,0);
        acc29 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc29,0,0,0);
        acc30 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc30,0,0,0);
        acc31 = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc31,0,0,0);

        acc0.x += val;   // prevents full optimization
    }

    float sum =
        acc0.x + acc1.x + acc2.x + acc3.x +
        acc4.x + acc5.x + acc6.x + acc7.x +
        acc8.x + acc9.x + acc10.x + acc11.x +
        acc12.x + acc13.x + acc14.x + acc15.x +
        acc16.x + acc17.x + acc18.x + acc19.x +
        acc20.x + acc21.x + acc22.x + acc23.x +
        acc24.x + acc25.x + acc26.x + acc27.x +
        acc28.x + acc29.x + acc30.x + acc31.x;

    out[gid] = sum;
}
