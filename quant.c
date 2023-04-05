#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK 32
#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct ggml_tensor {
  int n_dims;
  int64_t num_elements[4]; // number of elements
  size_t num_bytes[4];     // stride in bytes:
                           // nb[0] = sizeof(type)
                           // nb[1] = nb[0]   * ne[0] + padding
                           // nb[i] = nb[i-1] * ne[i-1]

  struct ggml_tensor *x;
  struct ggml_tensor *y;

  void *data;
};

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

struct ggml_compute_params {
  int ith;
  int nth;

  // work buffer for all threads
  void *wdata;
};

static void ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  const int64_t ne00 = src0->num_elements[0];
  const int64_t ne01 = src0->num_elements[1];
  const int64_t ne02 = src0->num_elements[2];
  const int64_t ne03 = src0->num_elements[3];

  const int64_t ne11 = src1->num_elements[1];
  const int64_t ne12 = src1->num_elements[2];

  const int64_t ne0 = dst->num_elements[0];

  const int nb01 = src0->num_bytes[1];
  const int nb02 = src0->num_bytes[2];
  const int nb03 = src0->num_bytes[3];

  const int nb0 = dst->num_bytes[0];
  const int nb1 = dst->num_bytes[1];
  const int nb2 = dst->num_bytes[2];
  const int nb3 = dst->num_bytes[3];

  const int ith = params->ith;
  const int nth = params->nth;

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

    void *src0_row =
        (void *)((char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    char *src1_col =
        ((char *)wdata + ((0 + i02 * ne11 + i03 * ne12 * ne11) * row_size));

    float *dst_col = (float *)((char *)dst->data +
                               (i01 * nb0 + 0 * nb1 + i02 * nb2 + i03 * nb3));

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
