#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void short_lived_acc(__global float* out)
{
    uint tid = get_local_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    float sum = 0.0f;

    for (int i = 0; i < 8; ++i)
    {
        float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
            a,
            b,
            acc,
            0, 0, 0
        );

        sum += acc.x + acc.y + acc.z + acc.w;
    }

    out[tid] = sum;
}
