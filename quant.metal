#ifdef __METAL__
#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;
#else

#define kernel
#define constant
#define device

typedef int int4[4];
typedef long long4[4];
typedef unsigned long ulong;

#define extract_bits(x, offset, count)                                         \
  (((x) >> (offset)) & ((1 << (count)) - 1))

#include <stdint.h>
#include <stdlib.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
  float d;        // delta
  uint8_t qs[16]; // nibbles / quants
} mps_block;

// n-dimensional tensor
struct mps_tensor {
  int4 size;
  int4 num_bytes;

  device void *data;
};

kernel void mps_ggml_compute_forward_mul_mat_q_f32(
    constant int *ith, constant int *nth, device char *wdata,
    device const struct mps_tensor *src0, device const struct mps_tensor *src1,
    device struct mps_tensor *dst) {

  // total rows in src0
  const int num_rows = src0->size[1] * src0->size[2] * src0->size[3];

  // rows per thread
  const int num_rows_per_thread = (num_rows + *nth - 1) / *nth;

  const int thread_id = *ith;

  // row range for this thread
  const int start_row = num_rows_per_thread * thread_id;

  const size_t row_size = src0->size[0] * sizeof(mps_block) / 32;

  for (int row_idx = start_row;
       row_idx < MIN(start_row + num_rows_per_thread, num_rows); ++row_idx) {
    // src0 indices

    auto src0_row =
        (device mps_block *)(src0->data + (row_idx * src0->num_bytes[1]));
    auto src1_col = wdata;

    auto dst_col = (device float *)(dst->data + (row_idx * dst->num_bytes[0]));

    for (int64_t col_idx = 0; col_idx < src1->size[1]; ++col_idx) {
      const auto x = (device mps_block *)src0_row;
      const auto y = (device mps_block *)&src1_col[col_idx * row_size];

      float sumf = 0.0;

      for (int i = 0; i < src0->size[0] / 32; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        for (int j = 0; j < 16; j++) {
          const uint8_t v0 = x[i].qs[j];
          const uint8_t v1 = y[i].qs[j];

          const float f0 = d0 * ((int8_t)(v0 & 0xf) - 8);
          const float f1 = d0 * ((int8_t)(v0 >> 4) - 8);

          const float f2 = d1 * ((int8_t)(v1 & 0xf) - 8);
          const float f3 = d1 * ((int8_t)(v1 >> 4) - 8);

          sumf += f0 * f2 + f1 * f3;
        }
      }

      dst_col[col_idx * dst->size[0]] = sumf;
    }
  }
}