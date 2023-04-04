#include "ggml.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK 32

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = {
    QK,
};

static_assert(GGML_TYPE_COUNT == 7, "GGML_TYPE_COUNT != 5");

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
    sizeof(block_q4_0),
};

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
  void *wdata;
};

static void ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *a,
    const struct ggml_tensor *b, struct ggml_tensor *dst) {

  const int nth = params->nth;
  const enum ggml_type type = a->type;
  if (type != GGML_TYPE_Q4_0) {
    abort();
  }

  // parallelize by x rows using ggml_vec_dot_q

  // total rows in a
  const int num_rows = a->ne[1] * a->ne[2] * a->ne[2];

  // rows per thread
  const int dr = (num_rows + nth - 1) / nth;

  // row range for this thread
  const int min_row = dr * params->ith;
  const int max_row = MIN(min_row + dr, num_rows);

  void *wdata = params->wdata;
  const size_t row_size = a->ne[0] * sizeof(block_q4_0) / QK;

  for (int row_index = min_row; row_index < max_row; ++row_index) {
    // a indices
    const int index3 = row_index / (a->ne[2] * a->ne[1]);
    const int index2 = (row_index - index3 * a->ne[2] * a->ne[1]) / a->ne[1];
    const int index1 =
        (row_index - index3 * a->ne[2] * a->ne[1] - index2 * a->ne[1]);

    void *x_row =
        (void *)((char *)a->data +
                 (index1 * a->nb[1] + index2 * a->nb[2] + index3 * a->nb[3]));
    char *y_col =
        ((char *)wdata +
         ((0 + index2 * b->ne[1] + index3 * b->ne[2] * b->ne[1]) * row_size));

    float *dest_column = (float *)((char *)dst->data +
                                   (index1 * dst->nb[0] + 0 * dst->nb[1] +
                                    index2 * dst->nb[2] + index3 * dst->nb[3]));

    for (int64_t ic = 0; ic < b->ne[1]; ++ic) {
      const int nb = a->ne[0] / QK;

      const block_q4_0 *restrict block_x = x_row;
      const block_q4_0 *restrict block_y = (void *)(y_col + ic * row_size);

      float sumf = 0.0;

      // scalar
      for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < QK / 2; j++) {
          const uint8_t v0 = block_x[i].qs[j];
          const uint8_t v1 = block_y[i].qs[j];

          const int8_t i0 = (int8_t)(v0 & 0xf) - 8;
          const int8_t i1 = (int8_t)(v0 >> 4) - 8;

          const int8_t i2 = (int8_t)(v1 & 0xf) - 8;
          const int8_t i3 = (int8_t)(v1 >> 4) - 8;

          sumi += i0 * i2 + i1 * i3;
        }
        sumf += block_x[i].d * block_y[i].d * sumi;
      }

      dest_column[ic * dst->ne[0]] = sumf;
    }
  }
}

static void impl(const struct ggml_compute_params *params,
                 const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                 struct ggml_tensor *dst) {
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const int nb00 = src0->nb[0];
  const int nb01 = src0->nb[1];
  const int nb02 = src0->nb[2];
  const int nb03 = src0->nb[3];

  const int nb10 = src1->nb[0];
  const int nb11 = src1->nb[1];
  const int nb12 = src1->nb[2];
  const int nb13 = src1->nb[3];

  const int nb0 = dst->nb[0];
  const int nb1 = dst->nb[1];
  const int nb2 = dst->nb[2];
  const int nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  assert(ne02 == ne12);
  assert(ne03 == ne13);
  assert(ne2 == ne12);
  assert(ne3 == ne13);

  const enum ggml_type type = src0->type;
  if (type != GGML_TYPE_Q4_0) {
    return;
  }

  // total rows in src0
  const int nr = ne01 * ne02 * ne03;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  void *wdata = params->wdata;
  const size_t row_size = ne00 * sizeof(block_q4_0) / QK;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    const int i13 = i03;
    const int i12 = i02;

    const int i0 = i01;
    const int i2 = i02;
    const int i3 = i03;

    void *src0_row =
        (void *)((char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    char *src1_col =
        ((char *)wdata + ((0 + i12 * ne11 + i13 * ne12 * ne11) * row_size));

    float *dst_col = (float *)((char *)dst->data +
                               (i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3));

    assert(ne00 % 32 == 0);

    for (int64_t ic = 0; ic < ne11; ++ic) {
      const int n = ne00;
      float *restrict s = &dst_col[ic * ne0];
      const void *restrict vx = src0_row;
      const void *restrict vy = (void *)(src1_col + ic * row_size);

      const int nb = n / QK;

      assert(n % QK == 0);
      assert(nb % 2 == 0);

      const block_q4_0 *restrict x = vx;
      const block_q4_0 *restrict y = vy;

      float sumf = 0.0;

      // scalar
      for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const uint8_t *restrict p0 = x[i].qs;
        const uint8_t *restrict p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK / 2; j++) {
          const uint8_t v0 = p0[j];
          const uint8_t v1 = p1[j];

          const int8_t i0 = (int8_t)(v0 & 0xf) - 8;
          const int8_t i1 = (int8_t)(v0 >> 4) - 8;

          const int8_t i2 = (int8_t)(v1 & 0xf) - 8;
          const int8_t i3 = (int8_t)(v1 >> 4) - 8;

          sumi += i0 * i2 + i1 * i3;
        }
        sumf += d0 * d1 * sumi;
      }

      *s = sumf;
    }
  }
}
