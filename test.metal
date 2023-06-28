#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float *a, device const float *b,
                       device float *result,
                       uint index [[thread_position_in_grid]]) {
  // the for-loop is replaced with a collection of threads, each of which
  // calls this function.
  result[index] = a[index] + b[index];
}
