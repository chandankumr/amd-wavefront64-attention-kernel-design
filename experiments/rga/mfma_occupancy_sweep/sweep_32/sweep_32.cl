#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sweep_32(__global float* out,
                       __global float* in)
{
    int gid = get_global_id(0);

    half4 a = (half4)(1.0h,1.0h,1.0h,1.0h);
    half4 b = (half4)(1.0h,1.0h,1.0h,1.0h);

    float4 acc[32];

    #pragma unroll
    for (int i = 0; i < 32; ++i)
        acc[i] = (float4)(0.0f);

    for (int iter = 0; iter < 32; ++iter)
    {
        float v = in[gid];

        #pragma unroll
        for (int i = 0; i < 32; ++i)
            acc[i] = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[i],0,0,0);

        #pragma unroll
        for (int i = 0; i < 32; ++i)
            acc[i].x += v;
    }

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; ++i)
        sum += acc[i].x;

    out[gid] = sum;
}
