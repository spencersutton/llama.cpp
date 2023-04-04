#include "ggml.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define QK 32

#define ne0(tensor) tensor->ne[0]
#define ne1(tensor) tensor->ne[1]
#define ne2(tensor) tensor->ne[2]
#define ne3(tensor) tensor->ne[3]

#define nb0(tensor) tensor->nb[0]
#define nb1(tensor) tensor->nb[1]
#define nb2(tensor) tensor->nb[2]
#define nb3(tensor) tensor->nb[3]

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

enum ggml_task_type {
  GGML_TASK_INIT = 0,
  GGML_TASK_COMPUTE,
  GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
  enum ggml_task_type type;

  int thread_index, num_threads;

  // work buffer for all threads
  size_t wsize;
  void *wdata;
};

static void ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *a,
    const struct ggml_tensor *b, struct ggml_tensor *dst) {

  const int num_threads = params->num_threads;
  const enum ggml_type type = a->type;
  if (type != GGML_TYPE_Q4_0) {
    abort();
  }

  // parallelize by x rows using ggml_vec_dot_q

  // total rows in a
  const int num_rows = ne1(a) * ne2(a) * ne2(a);

  // rows per thread
  const int dr = (num_rows + num_threads - 1) / num_threads;

  // row range for this thread
  const int min_row = dr * params->thread_index;
  const int max_row = MIN(min_row + dr, num_rows);

  void *wdata = params->wdata;
  const size_t row_size = ne0(a) * sizeof(block_q4_0) / QK;

  for (int row_index = min_row; row_index < max_row; ++row_index) {
    // a indices
    const int index3 = row_index / (ne2(a) * ne1(a));
    const int index2 = (row_index - index3 * ne2(a) * ne1(a)) / ne1(a);
    const int index1 = (row_index - index3 * ne2(a) * ne1(a) - index2 * ne1(a));

    void *x_row =
        (void *)((char *)a->data +
                 (index1 * nb1(a) + index2 * nb2(a) + index3 * nb3(a)));
    char *y_col =
        ((char *)wdata +
         ((0 + index2 * ne1(b) + index3 * ne2(b) * ne1(b)) * row_size));

    float *dest_column =
        (float *)((char *)dst->data + (index1 * nb0(dst) + 0 * nb1(dst) +
                                       index2 * nb2(dst) + index3 * nb3(dst)));

    for (int64_t ic = 0; ic < ne1(b); ++ic) {
      const int nb = ne0(a) / QK;

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

      dest_column[ic * ne0(dst)] = sumf;
    }
  }
}

static void impl() {
  // a indices
  const int index3 = row_index / (ne2(a) * ne1(a));
  const int index2 = (row_index - index3 * ne2(a) * ne1(a)) / ne1(a);
  const int index1 = (row_index - index3 * ne2(a) * ne1(a) - index2 * ne1(a));

  void *x_row = (void *)((char *)a->data +
                         (index1 * nb1(a) + index2 * nb2(a) + index3 * nb3(a)));
  char *y_col = ((char *)wdata +
                 ((0 + index2 * ne1(b) + index3 * ne2(b) * ne1(b)) * row_size));

  float *dest_column =
      (float *)((char *)dst->data + (index1 * nb0(dst) + 0 * nb1(dst) +
                                     index2 * nb2(dst) + index3 * nb3(dst)));

  for (int64_t ic = 0; ic < ne1(b); ++ic) {
    const int nb = ne0(a) / QK;

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

    dest_column[ic * ne0(dst)] = sumf;
  }
}