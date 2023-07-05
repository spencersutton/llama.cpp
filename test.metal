#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float *a, device float *result,
                       threadgroup float *sum,
                       uint siit [[simdgroup_index_in_threadgroup]],
                       uint spt [[simdgroups_per_threadgroup]],
                       uint tpt [[threads_per_threadgroup]],
                       uint tps [[threads_per_simdgroup]],
                       uint tiis [[thread_index_in_simdgroup]]) {
  float d = simd_sum(a[tiis + tps * siit]);
  sum[siit] = d;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < spt; i++)
    result[siit] += sum[i];
}
