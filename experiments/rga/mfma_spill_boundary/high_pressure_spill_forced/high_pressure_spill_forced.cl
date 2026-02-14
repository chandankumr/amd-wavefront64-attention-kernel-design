#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void high_pressure_spill_forced(__global float* out,
                                         __global float* in)
{
    int gid = get_global_id(0);

    half4 a = (half4)(1.0h, 1.0h, 1.0h, 1.0h);
    half4 b = (half4)(1.0h, 1.0h, 1.0h, 1.0h);

    // 64 independent persistent MFMA accumulators
    float4 acc[64];
    for (int i = 0; i < 64; ++i)
        acc[i] = (float4)(0.0f);

    // Persistent VGPR pressure
    float t0=0,t1=1,t2=2,t3=3,t4=4,t5=5,t6=6,t7=7;
    float t8=8,t9=9,t10=10,t11=11,t12=12,t13=13,t14=14,t15=15;
    float t16=16,t17=17,t18=18,t19=19,t20=20,t21=21,t22=22,t23=23;
    float t24=24,t25=25,t26=26,t27=27,t28=28,t29=29,t30=30,t31=31;

    for (int iter = 0; iter < 32; ++iter)
    {
        float v = in[gid + iter];

        // Increase lifetime pressure
        t0 += v;  t1 += t0;  t2 += t1;  t3 += t2;
        t4 += t3; t5 += t4;  t6 += t5;  t7 += t6;
        t8 += t7; t9 += t8;  t10 += t9; t11 += t10;
        t12 += t11; t13 += t12; t14 += t13; t15 += t14;
        t16 += t15; t17 += t16; t18 += t17; t19 += t18;
        t20 += t19; t21 += t20; t22 += t21; t23 += t22;
        t24 += t23; t25 += t24; t26 += t25; t27 += t26;
        t28 += t27; t29 += t28; t30 += t29; t31 += t30;

        // MFMA on all 64 accumulators
        for (int i = 0; i < 64; ++i)
        {
            acc[i] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        a, b, acc[i], 0, 0, 0);
        }
    }

    // Consume everything (avoid DCE)
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i)
        sum += acc[i].x;

    sum += t0+t1+t2+t3+t4+t5+t6+t7+
           t8+t9+t10+t11+t12+t13+t14+t15+
           t16+t17+t18+t19+t20+t21+t22+t23+
           t24+t25+t26+t27+t28+t29+t30+t31;

    out[gid] = sum;
}
