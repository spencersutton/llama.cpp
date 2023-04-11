#include "ggml.h"
#include <assert.h>
#include <stdint.h>

#define QK 32
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

enum ggml_task_type {
  GGML_TASK_INIT = 0,
  GGML_TASK_COMPUTE,
  GGML_TASK_FINALIZE,
};

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

struct ggml_compute_params {
  enum ggml_task_type type;

  int ith, nth;

  // work buffer for all threads
  size_t wsize;
  void *wdata;
};
static void ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    const struct ggml_tensor *src1, struct ggml_tensor *dst) {

  // total rows in src0
  const int num_rows = src0->size[1] * src0->size[2] * src0->size[3];

  // rows per thread
  const int num_rows_per_thread = (num_rows + params->nth - 1) / params->nth;

  const int thread_id = params->ith;

  // row range for this thread
  const int start_row = num_rows_per_thread * thread_id;

  const size_t row_size = src0->size[0] * sizeof(block_q4_0) / QK;

  for (int row_index = start_row;
       row_index < MIN(start_row + num_rows_per_thread, num_rows);
       ++row_index) {
    // src0 indices
    const int index3 = row_index / (src0->size[2] * src0->size[1]);
    const int index2 =
        (row_index - index3 * src0->size[2] * src0->size[1]) / src0->size[1];
    const int index1 = (row_index - index3 * src0->size[2] * src0->size[1] -
                        index2 * src0->size[1]);

  
    block_q4_0 *src0_row =
        (block_q4_0 *)((char *)src0->data +
                       (index1 * src0->nb[1] + index2 * src0->nb[2] +
                        index3 * src0->nb[3]));
    char *src1_col =
        ((char *)params->wdata + ((0 + index2 * src1->size[1] +
                                   index3 * src1->size[2] * src1->size[1]) *
                                  row_size));

    float *dst_col = (float *)((char *)dst->data +
                               (index1 * dst->nb[0] + 0 * dst->nb[1] +
                                index2 * dst->nb[2] + index3 * dst->nb[3]));

    assert(src0->size[0] % 32 == 0);

    for (int64_t column_index = 0; column_index < src1->size[1];
         ++column_index) {
      const int nb = src0->size[0] / QK;

      assert(src0->size[0] % QK == 0);
      assert(nb % 2 == 0);

      const block_q4_0 *restrict y =
          (void *)(src1_col + column_index * row_size);

      // scalar
      for (int i = 0; i < nb; i++) {
        const float d0 = src0_row[i].d;
        const float d1 = y[i].d;

        const uint8_t *restrict p0 = src0_row[i].qs;
        const uint8_t *restrict p1 = y[i].qs;

        for (int j = 0; j < QK / 2; j++) {
          const uint8_t v0 = p0[j];
          const uint8_t v1 = p1[j];

          const float f0 = d0 * ((int8_t)(v0 & 0xf) - 8);
          const float f1 = d0 * ((int8_t)(v0 >> 4) - 8);

          const float f2 = d1 * ((int8_t)(v1 & 0xf) - 8);
          const float f3 = d1 * ((int8_t)(v1 >> 4) - 8);

          dst_col[column_index * dst->size[0]] += f0 * f2 + f1 * f3;
        }
      }
    }
  }
}
