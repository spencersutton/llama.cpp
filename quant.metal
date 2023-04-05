#include <metal_atomic>
#include <metal_stdlib>

#ifndef __METAL__
#define kernel
#define device
typedef long long4[4];
typedef unsigned long ulong;
#define extract_bits(x, offset, count)                                         \
  (((x) >> (offset)) & ((1 << (count)) - 1))
#include <stdint.h>
#include <stdlib.h>
#endif

using namespace metal;

#define QK 32
#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct ggml_tensor {
  int n_dims;
  long4 ne; // number of elements
  long4 nb; // stride in bytes:
            // nb[0] = sizeof(type)
            // nb[1] = nb[0]   * ne[0] + padding
  // nb[i] = nb[i-1] * ne[i-1]

  device struct ggml_tensor *x;
  device struct ggml_tensor *y;

  device void *data;
};

typedef struct {
  float d;        // delta
  uint8_t qs[16]; // nibbles / quants
} block_q4_0;

struct ggml_compute_params {
  int ith;
  int nth;

  // work buffer for all threads
  device void *wdata;
};

kernel void ggml_compute_forward_mul_mat_q_f32(
    const device struct ggml_compute_params *params,
    const device struct ggml_tensor *src0,
    const device struct ggml_tensor *src1, device struct ggml_tensor *dst) {

  const int ith = params->ith;
  const int nth = params->nth;

  // total rows in src0
  const int nr = src0->ne[1] * src0->ne[2] * src0->ne[3];

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  auto wdata = params->wdata;
  const size_t row_size = src0->ne[0] * sizeof(block_q4_0) / QK;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (src0->ne[2] * src0->ne[1]);
    const int i02 = (ir - i03 * src0->ne[2] * src0->ne[1]) / src0->ne[1];
    const int i01 = (ir - i03 * src0->ne[2] * src0->ne[1] - i02 * src0->ne[1]);

    auto src0_row = (device char *)src0->data +
                    (i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
    auto src1_col =
        (device char *)wdata +
        ((0 + i02 * src1->ne[1] + i03 * src1->ne[2] * src1->ne[1]) * row_size);

    auto dst_col = (device float *)((device char *)dst->data +
                                    (i01 * dst->nb[0] + 0 * dst->nb[1] +
                                     i02 * dst->nb[2] + i03 * dst->nb[3]));

    for (int64_t ic = 0; ic < src1->ne[1]; ++ic) {
      const int nb = src0->ne[0] / QK;

      auto x = (device const block_q4_0 *)src0_row;
      auto y = (device const block_q4_0 *)(src1_col + ic * row_size);

      float sumf = 0.0;

      // scalar
      for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (ulong j = 0; j < sizeof(x[i].qs); j++) {
          const int v0 = x[i].qs[j];
          const int v1 = y[i].qs[j];

          const int i0 = extract_bits(v0, 4, 4) - 8;
          const int i1 = extract_bits(v0, 0, 4) - 8;

          const int i2 = extract_bits(v1, 4, 4) - 8;
          const int i3 = extract_bits(v1, 0, 4) - 8;

          sumi += i0 * i2 + i1 * i3;
        }
        sumf += x[i].d * y[i].d * sumi;
      }

      dst_col[ic * dst->ne[0]] = sumf;
    }
  }
}
