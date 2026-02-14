#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void memory_stress(__global float* out,
                            __global float* in)
{
    int gid = get_global_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    float4 acc = (float4)(0.0f);

    float tmp0, tmp1, tmp2, tmp3;
    float tmp4, tmp5, tmp6, tmp7;

    for (int i = 0; i < 64; ++i)
    {
        // Heavy global loads
        tmp0 = in[gid + i];
        tmp1 = in[gid + i + 128];
        tmp2 = in[gid + i + 256];
        tmp3 = in[gid + i + 384];

        tmp4 = tmp0 + tmp1;
        tmp5 = tmp2 + tmp3;
        tmp6 = tmp4 * tmp5;
        tmp7 = tmp6 + tmp0;

        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc,0,0,0);
        acc.x += tmp7;
    }

    out[gid] = acc.x;
}
