#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float *a, device const float *b,
                       device float *result,
                       uint threadGroupSize [[threads_per_threadgroup]],
                       uint simdSize [[threads_per_simdgroup]],
                       uint index [[thread_index_in_simdgroup]]) {
  float2 a2 = float2(a[index], b[index]);
  float c = a[index] + b[index];
  float d = simd_sum(a[index]);
  result[index] = d;
}
