#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void long_lived_acc(__global float* out)
{
    uint tid = get_local_id(0);

    // Proper half4 operands (required on CDNA4)
    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    // Accumulator lives across loop
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 8; ++i)
    {
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
            a,
            b,
            acc,
            0, 0, 0
        );
    }

    out[tid] = acc.x + acc.y + acc.z + acc.w;
}
