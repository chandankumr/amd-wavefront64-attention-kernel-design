// persistent_acc.cl
// Wave64 Persistent Accumulator Microkernel (CDNA / gfx950)

__kernel void persistent_mfma(
    __global float* A,
    __global float* B,
    __global float* C)
{
    const int lane = get_local_id(0);

    // Load per-lane inputs
    float a_frag = A[lane];
    float b_frag = B[lane];

    // Persistent accumulators (simulate HEAD_DIM = 64)
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    // Simulated K loop
    for (int k = 0; k < 64; k++)
    {
        acc0 += a_frag * b_frag;
        acc1 += a_frag * b_frag;
        acc2 += a_frag * b_frag;
        acc3 += a_frag * b_frag;
        acc4 += a_frag * b_frag;
        acc5 += a_frag * b_frag;
        acc6 += a_frag * b_frag;
        acc7 += a_frag * b_frag;
    }

    C[lane] = acc0 + acc1 + acc2 + acc3 +
              acc4 + acc5 + acc6 + acc7;
}