#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void acc_1(__global float* out)
{
    uint tid = get_local_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    float4 acc0 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 16; ++i)
    {
        acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(
            a, b,
            acc0,
            0, 0, 0
        );
    }

    out[tid] = acc0.x + acc0.y + acc0.z + acc0.w;
}
