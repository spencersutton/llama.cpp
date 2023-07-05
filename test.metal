#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float *a, device float *result,
                       threadgroup float *sum,
                       uint siit [[simdgroup_index_in_threadgroup]],
                       uint spt [[simdgroups_per_threadgroup]],
                       uint tpt [[threads_per_threadgroup]],
                       uint tps [[threads_per_simdgroup]],
                       uint tiis [[thread_index_in_simdgroup]],
                       uint tii [[thread_index_in_threadgroup]]) {
  float d = simd_sum(a[tiis + tps * siit]);
  sum[siit] = d;

  // threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tii == 0) {
    for (uint i = 0; i < spt; i++) {
      result[0] += sum[i];
    }
  }
}
