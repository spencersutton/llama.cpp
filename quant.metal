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

#define GGML_MAX_DIMS 4
#define GGML_MAX_OPT 4
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define QK 32

using namespace metal;

// available tensor operations:
enum ggml_op {
  GGML_OP_NONE = 0,

  GGML_OP_DUP,
  GGML_OP_ADD,
  GGML_OP_SUB,
  GGML_OP_MUL,
  GGML_OP_DIV,
  GGML_OP_SQR,
  GGML_OP_SQRT,
  GGML_OP_SUM,
  GGML_OP_MEAN,
  GGML_OP_REPEAT,
  GGML_OP_ABS,
  GGML_OP_SGN,
  GGML_OP_NEG,
  GGML_OP_STEP,
  GGML_OP_RELU,
  GGML_OP_GELU,
  GGML_OP_SILU,
  GGML_OP_NORM, // normalize
  GGML_OP_RMS_NORM,

  GGML_OP_MUL_MAT,

  GGML_OP_SCALE,
  GGML_OP_CPY,
  GGML_OP_CONT,
  GGML_OP_RESHAPE,
  GGML_OP_VIEW,
  GGML_OP_PERMUTE,
  GGML_OP_TRANSPOSE,
  GGML_OP_GET_ROWS,
  GGML_OP_DIAG_MASK_INF,
  GGML_OP_SOFT_MAX,
  GGML_OP_ROPE,
  GGML_OP_CONV_1D_1S,
  GGML_OP_CONV_1D_2S,

  GGML_OP_FLASH_ATTN,
  GGML_OP_FLASH_FF,

  GGML_OP_COUNT,
};

enum ggml_task_type {
  GGML_TASK_INIT = 0,
  GGML_TASK_COMPUTE,
  GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
  enum ggml_task_type type;

  int ith, nth;

  // work buffer for all threads
  size_t wsize;
  device void *wdata;
};

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

enum ggml_type {
  // explicitly numbered values are used in llama.cpp files
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_I8,
  GGML_TYPE_I16,
  GGML_TYPE_I32,
  GGML_TYPE_COUNT,
};

// n-dimensional tensor
struct ggml_tensor {
  enum ggml_type type;

  int n_dims;
  int64_t size[GGML_MAX_DIMS]; // number of elements
  size_t nb[GGML_MAX_DIMS];    // stride in bytes:
                               // nb[0] = sizeof(type)
                               // nb[1] = nb[0]   * ne[0] + padding
                               // nb[i] = nb[i-1] * ne[i-1]

  // compute data
  enum ggml_op op;

  bool is_param;

  device struct ggml_tensor *grad;
  device struct ggml_tensor *src0;
  device struct ggml_tensor *src1;
  device struct ggml_tensor *opt[GGML_MAX_OPT];

  // thread scheduling
  int n_tasks;

  // performance
  int perf_runs;
  int64_t perf_cycles;
  int64_t perf_time_us;

  device void *data;
  char padding[8];
};

void ggml_compute_forward_mul_mat_q_f32(
    device const struct ggml_compute_params *params,
    device const struct ggml_tensor *src0,
    device const struct ggml_tensor *src1, device struct ggml_tensor *dst) {

  assert(params->type != GGML_TASK_INIT);
  assert(params->type != GGML_TASK_FINALIZE);

  // parallelize by src0 rows using ggml_vec_dot_q

  // total rows in src0
  const int num_rows = src0->size[1] * src0->size[2] * src0->size[3];

  // rows per thread
  const int num_rows_per_thread = (num_rows + params->nth - 1) / params->nth;

  const int thread_id = params->ith;

  // row range for this thread
  const int start_row = num_rows_per_thread * thread_id;

  const size_t row_size = src0->size[0] * sizeof(block_q4_0) / QK;

  for (int row_idx = start_row;
       row_idx < MIN(start_row + num_rows_per_thread, num_rows); ++row_idx) {
    // src0 indices

    device block_q4_0 *src0_row =
        (device block_q4_0 *)((device char *)src0->data +
                              (row_idx * src0->nb[1]));
    device char *src1_col = (device char *)params->wdata;

    device float *dst_col =
        (device float *)((device char *)dst->data + (row_idx * dst->nb[0]));

    for (int64_t col_idx = 0; col_idx < src1->size[1]; ++col_idx) {
      const auto x = (device block_q4_0 *)src0_row;
      const auto y = (device block_q4_0 *)&src1_col[col_idx * row_size];

      float sumf = 0.0;

      for (int i = 0; i < src0->size[0] / QK; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        for (int j = 0; j < QK / 2; j++) {
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