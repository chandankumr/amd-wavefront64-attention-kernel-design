#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void low_pressure(__global float* out,
                           __global float* in)
{
    int gid = get_global_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 8; i++)
    {
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                a,
                b,
                acc,
                0,
                0,
                0);

        float val = in[gid];   // global load
        acc.x += val;
    }

    out[gid] = acc.x;
}
