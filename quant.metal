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

struct ggml_compute_params {
  int ith;
  int nth;

  // work buffer for all threads
  device char *wdata;
};

typedef struct {
  float d;        // delta
  uint8_t qs[16]; // nibbles / quants
} block_q4_0;

// n-dimensional tensor
struct ggml_tensor {
  int4 size;
  int4 num_bytes;

  device char *data;
};

kernel void ggml_compute_forward_mul_mat_q_f32(
    constant int *ith, constant int *nth, device char *wdata,
    device const struct ggml_tensor *src0,
    device const struct ggml_tensor *src1, device struct ggml_tensor *dst) {

  // parallelize by src0 rows using ggml_vec_dot_q

  // total rows in src0
  const int num_rows = src0->size[1] * src0->size[2] * src0->size[3];

  // rows per thread
  const int num_rows_per_thread = (num_rows + *nth - 1) / *nth;

  const int thread_id = *ith;

  // row range for this thread
  const int start_row = num_rows_per_thread * thread_id;

  const size_t row_size = src0->size[0] * sizeof(block_q4_0) / 32;

  for (int row_idx = start_row;
       row_idx < MIN(start_row + num_rows_per_thread, num_rows); ++row_idx) {
    // src0 indices

    auto src0_row =
        (device block_q4_0 *)(src0->data + (row_idx * src0->num_bytes[1]));
    auto src1_col = wdata;

    auto dst_col = (device float *)(dst->data + (row_idx * dst->num_bytes[0]));

    for (int64_t col_idx = 0; col_idx < src1->size[1]; ++col_idx) {
      const auto x = (device block_q4_0 *)src0_row;
      const auto y = (device block_q4_0 *)&src1_col[col_idx * row_size];

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