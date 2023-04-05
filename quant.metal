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
  int64_t ne[4]; // number of elements
  size_t nb[4];  // stride in bytes:
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
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];

  const int64_t ne0 = dst->ne[0];

  const int nb01 = src0->nb[1];
  const int nb02 = src0->nb[2];
  const int nb03 = src0->nb[3];

  const int nb0 = dst->nb[0];
  const int nb1 = dst->nb[1];
  const int nb2 = dst->nb[2];
  const int nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  // total rows in src0
  const int nr = ne01 * ne02 * ne03;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  auto wdata = params->wdata;
  const size_t row_size = ne00 * sizeof(block_q4_0) / QK;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    auto src0_row =
        (device char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03);
    auto src1_col = (device char *)wdata +
                    ((0 + i02 * ne11 + i03 * ne12 * ne11) * row_size);

    auto dst_col =
        (device float *)((device char *)dst->data +
                         (i01 * nb0 + 0 * nb1 + i02 * nb2 + i03 * nb3));

    for (int64_t ic = 0; ic < ne11; ++ic) {
      const int nb = ne00 / QK;

      auto x = (device const block_q4_0 *)src0_row;
      auto y = (device const block_q4_0 *)(src1_col + ic * row_size);

      float sumf = 0.0;

      // scalar
      for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (ulong j = 0; j < sizeof(x[i].qs); j++) {
          const auto v0 = x[i].qs[j];
          const auto v1 = y[i].qs[j];

          const auto i0 = extract_bits(v0, 4, 4) - 8;
          const auto i1 = extract_bits(v0, 0, 4) - 8;

          const auto i2 = extract_bits(v1, 4, 4) - 8;
          const auto i3 = extract_bits(v1, 0, 4) - 8;

          sumi += i0 * i2 + i1 * i3;
        }
        sumf += x[i].d * y[i].d * sumi;
      }

      dst_col[ic * ne0] = sumf;
    }
  }
}
