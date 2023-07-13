#include "k_quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#include "ggml.h"

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
  assert(fval <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float *restrict x, int8_t *restrict L, int rmse_type) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (!amax) {  // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
    }
    return 1 / iscale;
  }
  int weight_type = rmse_type % 2;
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
    float w = weight_type == 1 ? x[i] * x[i] : 1;
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = sumlx / suml2;
  float best = scale * sumlx;
  for (int itry = 0; itry < 3; ++itry) {
    iscale = 1 / scale;
    float slx = 0;
    float sl2 = 0;
    bool changed = false;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      if (l + nmax != L[i]) {
        changed = true;
      }
      float w = weight_type == 1 ? x[i] * x[i] : 1.f;
      slx += w * x[i] * l;
      sl2 += w * l * l;
    }
    if (!changed || sl2 == 0 || slx * slx <= best * sl2) {
      break;
    }
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
    }
    sumlx = slx;
    suml2 = sl2;
    scale = sumlx / suml2;
    best = scale * sumlx;
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      float w = weight_type == 1 ? x[i] * x[i] : 1;
      int l = L[i] - nmax;
      float slx = sumlx - w * x[i] * l;
      if (slx > 0) {
        float sl2 = suml2 - w * l * l;
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = MAX(-nmax, MIN(nmax - 1, new_l));
        if (new_l != l) {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = nmax + new_l;
            sumlx = slx;
            suml2 = sl2;
            scale = sumlx / suml2;
            best = scale * sumlx;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) {
      break;
    }
  }
  if (rmse_type < 3) {
    return scale;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0) {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      float w = weight_type == 1 ? x[i] * x[i] : 1;
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static float make_q3_quants(int n, int nmax, const float *restrict x, int8_t *restrict L, bool do_rmse) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (!amax) {  // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (do_rmse) {
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      L[i] = l;
      float w = x[i] * x[i];
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
      int n_changed = 0;
      for (int i = 0; i < n; ++i) {
        float w = x[i] * x[i];
        float slx = sumlx - w * x[i] * L[i];
        if (slx > 0) {
          float sl2 = suml2 - w * L[i] * L[i];
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = MAX(-nmax, MIN(nmax - 1, new_l));
          if (new_l != L[i]) {
            slx += w * x[i] * new_l;
            sl2 += w * new_l * new_l;
            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
              L[i] = new_l;
              sumlx = slx;
              suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed) {
        break;
      }
    }
    for (int i = 0; i < n; ++i) {
      L[i] += nmax;
    }
    return sumlx / suml2;
  }
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
  }
  return 1 / iscale;
}

static float make_qkx1_quants(int n, int nmax, const float *restrict x, uint8_t *restrict L, float *restrict the_min,
                              int ntry) {
  float min = x[0];
  float max = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
  }
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = 0;
    return 0.f;
  }
  if (min > 0) min = 0;
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  for (int itry = 0; itry < ntry; ++itry) {
    float sumlx = 0;
    int suml2 = 0;
    bool did_change = false;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      if (l != L[i]) {
        L[i] = l;
        did_change = true;
      }
      sumlx += (x[i] - min) * l;
      suml2 += l * l;
    }
    scale = sumlx / suml2;
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += x[i] - scale * L[i];
    }
    min = sum / n;
    if (min > 0) min = 0;
    iscale = 1 / scale;
    if (!did_change) break;
  }
  *the_min = -min;
  return scale;
}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t *restrict q, uint8_t *restrict d, uint8_t *restrict m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
#endif

//========================- 2-bit (de)-quantization

void quantize_row_q2_K_reference(const float *restrict x, block_q2_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  float mins[QK_K / 16];
  float scales[QK_K / 16];

  const float q4scale = 15.f;

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      scales[j] = make_qkx1_quants(16, 3, x + 16 * j, L + 16 * j, &mins[j], 5);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    if (max_scale > 0) {
      float iscale = q4scale / max_scale;
      for (int j = 0; j < QK_K / 16; ++j) {
        int l = nearest_int(iscale * scales[j]);
        y[i].scales[j] = l;
      }
      y[i].d = ggml_fp32_to_fp16(max_scale / q4scale);
    } else {
      for (int j = 0; j < QK_K / 16; ++j) y[i].scales[j] = 0;
      y[i].d = ggml_fp32_to_fp16(0.f);
    }
    if (max_min > 0) {
      float iscale = q4scale / max_min;
      for (int j = 0; j < QK_K / 16; ++j) {
        int l = nearest_int(iscale * mins[j]);
        y[i].scales[j] |= (l << 4);
      }
      y[i].dmin = ggml_fp32_to_fp16(max_min / q4scale);
    } else {
      y[i].dmin = ggml_fp32_to_fp16(0.f);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      const float d = ggml_fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
      if (!d) continue;
      const float dm = ggml_fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int((x[16 * j + ii] + dm) / d);
        l = MAX(0, MIN(3, l));
        L[16 * j + ii] = l;
      }
    }

#if QK_K == 256
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }
#else
    for (int l = 0; l < 16; ++l) {
      y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
    }
#endif

    x += QK_K;
  }
}

void dequantize_row_q2_K(const block_q2_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = ggml_fp16_to_fp32(x[i].d);
    const float min = ggml_fp16_to_fp32(x[i].dmin);

    const uint8_t *q = x[i].qs;

#if QK_K == 256
    int is = 0;
    float dl, ml;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

        sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

        shift += 2;
      }
      q += 32;
    }
#else
    float dl1 = d * (x[i].scales[0] & 0xF), ml1 = min * (x[i].scales[0] >> 4);
    float dl2 = d * (x[i].scales[1] & 0xF), ml2 = min * (x[i].scales[1] >> 4);
    float dl3 = d * (x[i].scales[2] & 0xF), ml3 = min * (x[i].scales[2] >> 4);
    float dl4 = d * (x[i].scales[3] & 0xF), ml4 = min * (x[i].scales[3] >> 4);
    for (int l = 0; l < 16; ++l) {
      y[l + 0] = dl1 * ((int8_t)((q[l] >> 0) & 3)) - ml1;
      y[l + 16] = dl2 * ((int8_t)((q[l] >> 2) & 3)) - ml2;
      y[l + 32] = dl3 * ((int8_t)((q[l] >> 4) & 3)) - ml3;
      y[l + 48] = dl4 * ((int8_t)((q[l] >> 6) & 3)) - ml4;
    }
    y += QK_K;
#endif
  }
}

void quantize_row_q2_K(const float *restrict x, void *restrict vy, int k) { quantize_row_q2_K_reference(x, vy, k); }

size_t ggml_quantize_q2_K(const float *restrict src, void *restrict dst, int n, int k, int64_t *restrict hist) {
  const int nb = k / QK_K;

  // TODO - collect histograms - although, at a second thought, I don't really care about them
  (void)hist;

  for (int j = 0; j < nb; j += k) {
    block_q2_K *restrict y = (block_q2_K *)dst + j / QK_K;
    quantize_row_q2_K_reference(src + j, y, k);
  }
  return (n / QK_K * sizeof(block_q2_K));
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K_reference(const float *restrict x, block_q3_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
      float scale = fabsf(scales[j]);
      if (scale > amax) {
        amax = scale;
        max_scale = scales[j];
      }
    }

#if QK_K == 256
    memset(y[i].scales, 0, 12);
    if (max_scale) {
      float iscale = -32.f / max_scale;
      for (int j = 0; j < QK_K / 16; ++j) {
        int8_t l = nearest_int(iscale * scales[j]);
        l = MAX(-32, MIN(31, l)) + 32;
        if (j < 8) {
          y[i].scales[j] = l & 0xF;
        } else {
          y[i].scales[j - 8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
      }
      y[i].d = ggml_fp32_to_fp16(1 / iscale);
    } else {
      y[i].d = ggml_fp32_to_fp16(0.f);
    }

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j) {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = ggml_fp16_to_fp32(y[i].d) * sc;
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-4, MIN(3, l));
        L[16 * j + ii] = l + 4;
      }
    }
#else
    if (max_scale) {
      float iscale = -8.f / max_scale;
      for (int j = 0; j < QK_K / 16; j += 2) {
        int l1 = nearest_int(iscale * scales[j]);
        l1 = 8 + MAX(-8, MIN(7, l1));
        int l2 = nearest_int(iscale * scales[j + 1]);
        l2 = 8 + MAX(-8, MIN(7, l2));
        y[i].scales[j / 2] = l1 | (l2 << 4);
      }
      y[i].d = ggml_fp32_to_fp16(1 / iscale);
    } else {
      for (int j = 0; j < QK_K / 16; j += 2) {
        y[i].scales[j / 2] = 0;
      }
      y[i].d = ggml_fp32_to_fp16(0.f);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int s = j % 2 == 0 ? y[i].scales[j / 2] & 0xF : y[i].scales[j / 2] >> 4;
      float d = ggml_fp16_to_fp32(y[i].d) * (s - 8);
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-4, MIN(3, l));
        L[16 * j + ii] = l + 4;
      }
    }
#endif

    memset(y[i].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {
      if (L[j] > 3) {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
      }
      if (++m == QK_K / 8) {
        m = 0;
        hm <<= 1;
      }
    }
#if QK_K == 256
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }
#else
    for (int l = 0; l < 16; ++l) {
      y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
    }
#endif

    x += QK_K;
  }
}

#if QK_K == 256
void dequantize_row_q3_K(const block_q3_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  uint32_t aux[4];
  const int8_t *scales = (const int8_t *)aux;

  for (int i = 0; i < nb; i++) {
    const float d_all = ggml_fp16_to_fp32(x[i].d);

    const uint8_t *restrict q = x[i].qs;
    const uint8_t *restrict hm = x[i].hmask;
    uint8_t m = 1;

    memcpy(aux, x[i].scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int is = 0;
    float dl;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
        }

        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }

        shift += 2;
        m <<= 1;
      }
      q += 32;
    }
  }
}
#else
void dequantize_row_q3_K(const block_q3_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  assert(QK_K == 64);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d_all = ggml_fp16_to_fp32(x[i].d);

    const uint8_t *restrict q = x[i].qs;
    const uint8_t *restrict hm = x[i].hmask;

    const float d1 = d_all * ((x[i].scales[0] & 0xF) - 8);
    const float d2 = d_all * ((x[i].scales[0] >> 4) - 8);
    const float d3 = d_all * ((x[i].scales[1] & 0xF) - 8);
    const float d4 = d_all * ((x[i].scales[1] >> 4) - 8);

    for (int l = 0; l < 8; ++l) {
      uint8_t h = hm[l];
      y[l + 0] = d1 * ((int8_t)((q[l + 0] >> 0) & 3) - ((h & 0x01) ? 0 : 4));
      y[l + 8] = d1 * ((int8_t)((q[l + 8] >> 0) & 3) - ((h & 0x02) ? 0 : 4));
      y[l + 16] = d2 * ((int8_t)((q[l + 0] >> 2) & 3) - ((h & 0x04) ? 0 : 4));
      y[l + 24] = d2 * ((int8_t)((q[l + 8] >> 2) & 3) - ((h & 0x08) ? 0 : 4));
      y[l + 32] = d3 * ((int8_t)((q[l + 0] >> 4) & 3) - ((h & 0x10) ? 0 : 4));
      y[l + 40] = d3 * ((int8_t)((q[l + 8] >> 4) & 3) - ((h & 0x20) ? 0 : 4));
      y[l + 48] = d4 * ((int8_t)((q[l + 0] >> 6) & 3) - ((h & 0x40) ? 0 : 4));
      y[l + 56] = d4 * ((int8_t)((q[l + 8] >> 6) & 3) - ((h & 0x80) ? 0 : 4));
    }
    y += QK_K;
  }
}
#endif

void quantize_row_q3_K(const float *restrict x, void *restrict vy, int k) { quantize_row_q3_K_reference(x, vy, k); }

size_t ggml_quantize_q3_K(const float *restrict src, void *restrict dst, int n, int k, int64_t *restrict hist) {
  const int nb = k / QK_K;

  // TODO - collect histograms - although, at a second thought, I don't really care about them
  (void)hist;

  for (int j = 0; j < nb; j += k) {
    block_q3_K *restrict y = (block_q3_K *)dst + j / QK_K;
    quantize_row_q3_K_reference(src + j, y, k);
  }
  return (n / QK_K * sizeof(block_q3_K));
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K_reference(const float *restrict x, block_q4_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      scales[j] = make_qkx1_quants(32, 15, x + 32 * j, L + 32 * j, &mins[j], 5);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

#if QK_K == 256
    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = ggml_fp32_to_fp16(max_scale / 63.f);
    y[i].dmin = ggml_fp32_to_fp16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = ggml_fp16_to_fp32(y[i].d) * sc;
      if (!d) continue;
      const float dm = ggml_fp16_to_fp32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
#else
    const float s_factor = 15.f;
    float inv_scale = max_scale > 0 ? s_factor / max_scale : 0.f;
    float inv_min = max_min > 0 ? s_factor / max_min : 0.f;
    int d1 = nearest_int(inv_scale * scales[0]);
    int m1 = nearest_int(inv_min * mins[0]);
    int d2 = nearest_int(inv_scale * scales[1]);
    int m2 = nearest_int(inv_min * mins[1]);
    y[i].scales[0] = d1 | (m1 << 4);
    y[i].scales[1] = d2 | (m2 << 4);
    y[i].d[0] = ggml_fp32_to_fp16(max_scale / s_factor);
    y[i].d[1] = ggml_fp32_to_fp16(max_min / s_factor);

    float sumlx = 0;
    int suml2 = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      const uint8_t sd = y[i].scales[j] & 0xF;
      const uint8_t sm = y[i].scales[j] >> 4;
      const float d = ggml_fp16_to_fp32(y[i].d[0]) * sd;
      if (!d) continue;
      const float m = ggml_fp16_to_fp32(y[i].d[1]) * sm;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + m) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
        sumlx += (x[32 * j + ii] + m) * l * sd;
        suml2 += l * l * sd * sd;
      }
    }
    if (suml2) {
      y[i].d[0] = ggml_fp32_to_fp16(sumlx / suml2);
    }
#endif
    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

void dequantize_row_q4_K(const block_q4_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t *q = x[i].qs;

#if QK_K == 256

    const float d = ggml_fp16_to_fp32(x[i].d);
    const float min = ggml_fp16_to_fp32(x[i].dmin);

    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
#else
    const float dall = ggml_fp16_to_fp32(x[i].d[0]);
    const float mall = ggml_fp16_to_fp32(x[i].d[1]);
    const float d1 = dall * (x[i].scales[0] & 0xF), m1 = mall * (x[i].scales[0] >> 4);
    const float d2 = dall * (x[i].scales[1] & 0xF), m2 = mall * (x[i].scales[1] >> 4);
    for (int l = 0; l < 32; ++l) {
      y[l + 0] = d1 * (q[l] & 0xF) - m1;
      y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
    y += QK_K;
#endif
  }
}

void quantize_row_q4_K(const float *restrict x, void *restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q4_K *restrict y = vy;
  quantize_row_q4_K_reference(x, y, k);
}

size_t ggml_quantize_q4_K(const float *restrict src, void *restrict dst, int n, int k, int64_t *restrict hist) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;
  (void)hist;  // TODO: collect histograms
  for (int j = 0; j < nb; j += k) {
    block_q4_K *restrict y = (block_q4_K *)dst + j / QK_K;
    quantize_row_q4_K_reference(src + j, y, k);
  }
  return (n / QK_K * sizeof(block_q4_K));
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K_reference(const float *restrict x, block_q5_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

#if QK_K == 256
  uint8_t L[QK_K];
  float mins[QK_K / 32];
  float scales[QK_K / 32];
#else
  int8_t L[QK_K];
  float scales[QK_K / 16];
#endif

  for (int i = 0; i < nb; i++) {
#if QK_K == 256

    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      scales[j] = make_qkx1_quants(32, 31, x + 32 * j, L + 32 * j, &mins[j], 5);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = ggml_fp32_to_fp16(max_scale / 63.f);
    y[i].dmin = ggml_fp32_to_fp16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = ggml_fp16_to_fp32(y[i].d) * sc;
      if (!d) continue;
      const float dm = ggml_fp16_to_fp32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(31, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *restrict qh = y[i].qh;
    uint8_t *restrict ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64) {
      for (int j = 0; j < 32; ++j) {
        int l1 = L[n + j];
        if (l1 > 15) {
          l1 -= 16;
          qh[j] |= m1;
        }
        int l2 = L[n + j + 32];
        if (l2 > 15) {
          l2 -= 16;
          qh[j] |= m2;
        }
        ql[j] = l1 | (l2 << 4);
      }
      m1 <<= 2;
      m2 <<= 2;
      ql += 32;
    }
#else
    float max_scale = 0, amax = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      scales[j] = make_qx_quants(16, 16, x + 16 * j, L + 16 * j, 1);
      float abs_scale = fabsf(scales[j]);
      if (abs_scale > amax) {
        amax = abs_scale;
        max_scale = scales[j];
      }
    }

    float iscale = -128.f / max_scale;
    for (int j = 0; j < QK_K / 16; ++j) {
      int l = nearest_int(iscale * scales[j]);
      y[i].scales[j] = MAX(-128, MIN(127, l));
    }
    y[i].d = ggml_fp32_to_fp16(1 / iscale);

    for (int j = 0; j < QK_K / 16; ++j) {
      const float d = ggml_fp16_to_fp32(y[i].d) * y[i].scales[j];
      if (!d) continue;
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-16, MIN(15, l));
        L[16 * j + ii] = l + 16;
      }
    }

    uint8_t *restrict qh = y[i].qh;
    uint8_t *restrict ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    for (int j = 0; j < 32; ++j) {
      int jm = j % 8;
      int is = j / 8;
      int l1 = L[j];
      if (l1 > 15) {
        l1 -= 16;
        qh[jm] |= (1 << is);
      }
      int l2 = L[j + 32];
      if (l2 > 15) {
        l2 -= 16;
        qh[jm] |= (1 << (4 + is));
      }
      ql[j] = l1 | (l2 << 4);
    }
#endif

    x += QK_K;
  }
}

void dequantize_row_q5_K(const block_q5_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t *ql = x[i].qs;
    const uint8_t *qh = x[i].qh;

#if QK_K == 256

    const float d = ggml_fp16_to_fp32(x[i].d);
    const float min = ggml_fp16_to_fp32(x[i].dmin);

    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
#else
    float d = ggml_fp16_to_fp32(x[i].d);
    const int8_t *restrict s = x[i].scales;
    for (int l = 0; l < 8; ++l) {
      y[l + 0] = d * s[0] * ((ql[l + 0] & 0xF) - (qh[l] & 0x01 ? 0 : 16));
      y[l + 8] = d * s[0] * ((ql[l + 8] & 0xF) - (qh[l] & 0x02 ? 0 : 16));
      y[l + 16] = d * s[1] * ((ql[l + 16] & 0xF) - (qh[l] & 0x04 ? 0 : 16));
      y[l + 24] = d * s[1] * ((ql[l + 24] & 0xF) - (qh[l] & 0x08 ? 0 : 16));
      y[l + 32] = d * s[2] * ((ql[l + 0] >> 4) - (qh[l] & 0x10 ? 0 : 16));
      y[l + 40] = d * s[2] * ((ql[l + 8] >> 4) - (qh[l] & 0x20 ? 0 : 16));
      y[l + 48] = d * s[3] * ((ql[l + 16] >> 4) - (qh[l] & 0x40 ? 0 : 16));
      y[l + 56] = d * s[3] * ((ql[l + 24] >> 4) - (qh[l] & 0x80 ? 0 : 16));
    }
    y += QK_K;
#endif
  }
}

void quantize_row_q5_K(const float *restrict x, void *restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q5_K *restrict y = vy;
  quantize_row_q5_K_reference(x, y, k);
}

size_t ggml_quantize_q5_K(const float *restrict src, void *restrict dst, int n, int k, int64_t *restrict hist) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;
  (void)hist;
  for (int j = 0; j < nb; j += k) {
    block_q5_K *restrict y = (block_q5_K *)dst + j / QK_K;
    quantize_row_q5_K_reference(src + j, y, k);
  }
  return (n / QK_K * sizeof(block_q5_K));
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_reference(const float *restrict x, block_q6_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
      const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1);
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    float iscale = -128.f / max_scale;
    y[i].d = ggml_fp32_to_fp16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = ggml_fp16_to_fp32(y[i].d) * y[i].scales[j];
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *restrict ql = y[i].ql;
    uint8_t *restrict qh = y[i].qh;
#if QK_K == 256
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] =
            (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }
#else
    for (int l = 0; l < 32; ++l) {
      const uint8_t q1 = L[l + 0] & 0xF;
      const uint8_t q2 = L[l + 32] & 0xF;
      ql[l] = q1 | (q2 << 4);
    }
    for (int l = 0; l < 16; ++l) {
      qh[l] = (L[l] >> 4) | ((L[l + 16] >> 4) << 2) | ((L[l + 32] >> 4) << 4) | ((L[l + 48] >> 4) << 6);
    }
#endif

    x += QK_K;
  }
}

void dequantize_row_q6_K(const block_q6_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = ggml_fp16_to_fp32(x[i].d);

    const uint8_t *restrict ql = x[i].ql;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict sc = x[i].scales;

#if QK_K == 256
    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
#else
    for (int l = 0; l < 16; ++l) {
      const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = (int8_t)((ql[l + 16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = (int8_t)((ql[l + 16] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[0] * q1;
      y[l + 16] = d * sc[1] * q2;
      y[l + 32] = d * sc[2] * q3;
      y[l + 48] = d * sc[3] * q4;
    }
    y += 64;
#endif
  }
}

void quantize_row_q6_K(const float *restrict x, void *restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q6_K *restrict y = vy;
  quantize_row_q6_K_reference(x, y, k);
}

size_t ggml_quantize_q6_K(const float *src, void *dst, int n, int k, int64_t *hist) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  (void)hist;  // TODO

  for (int j = 0; j < nb; j += k) {
    block_q6_K *restrict y = (block_q6_K *)dst + j / QK_K;
    quantize_row_q6_K_reference(src + j, y, k);
  }
  return (n / QK_K * sizeof(block_q6_K));
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_reference(const float *restrict x, block_q8_K *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    const float iscale = -128.f / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

void dequantize_row_q8_K(const block_q8_K *restrict x, float *restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) {
      *y++ = x[i].d * x[i].qs[j];
    }
  }
}

void quantize_row_q8_K(const float *restrict x, void *restrict y, int k) { quantize_row_q8_K_reference(x, y, k); }

//===================================== Dot ptoducts =================================

//
// Helper functions
//

#if QK_K == 256
void ggml_vec_dot_q2_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const block_q2_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  const uint8x16_t m3 = vdupq_n_u8(0x3);
  const uint8x16_t m4 = vdupq_n_u8(0xF);
  const int32x4_t vzero = vdupq_n_s32(0);

  int8x16x2_t q2bytes;
  uint8_t aux[16];

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
    const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

    const uint8_t *restrict q2 = x[i].qs;
    const int8_t *restrict q8 = y[i].qs;
    const uint8_t *restrict sc = x[i].scales;

    const uint8x16_t mins_and_scales = vld1q_u8(sc);
    const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
    vst1q_u8(aux, scales);

    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
    const int16x8x2_t mins16 = {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
                                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))};
    const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
                                   vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
    const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
                                   vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
    sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

    int isum = 0;
    int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)                                                    \
  isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is + (index)]; \
  isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is + 1 + (index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)                                     \
  q8bytes = vld1q_s8_x2(q8);                                                              \
  q8 += 32;                                                                               \
  q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3)); \
  q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3)); \
  MULTIPLY_ACCUM_WITH_SCALE((index));

    for (int j = 0; j < QK_K / 128; ++j) {
      const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
      q2 += 32;

      int8x16x2_t q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));
      MULTIPLY_ACCUM_WITH_SCALE(0);

      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);

      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);

      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

      is += 8;
    }
    sum += d * isum;
  }

  *s = sum;
}

#else

void ggml_vec_dot_q2_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const block_q2_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  const uint8x16_t m3 = vdupq_n_u8(0x3);
  const int32x4_t vzero = vdupq_n_s32(0);

  int8x16x4_t q2bytes;

  uint32_t aux32[2];
  const uint8_t *scales = (const uint8_t *)aux32;

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * (float)x[i].d;
    const float dmin = -y[i].d * (float)x[i].dmin;

    const uint8_t *restrict q2 = x[i].qs;
    const int8_t *restrict q8 = y[i].qs;
    const uint32_t *restrict sc = (const uint32_t *)x[i].scales;

    aux32[0] = sc[0] & 0x0f0f0f0f;
    aux32[1] = (sc[0] >> 4) & 0x0f0f0f0f;

    sum += dmin * (scales[4] * y[i].bsums[0] + scales[5] * y[i].bsums[1] + scales[6] * y[i].bsums[2] +
                   scales[7] * y[i].bsums[3]);

    int isum1 = 0, isum2 = 0;

    const uint8x16_t q2bits = vld1q_u8(q2);

    const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

    q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits, m3));
    q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 2), m3));
    q2bytes.val[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 4), m3));
    q2bytes.val[3] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 6), m3));

    isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * scales[0];
    isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * scales[1];
    isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[2], q8bytes.val[2])) * scales[2];
    isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[3], q8bytes.val[3])) * scales[3];
    sum += d * (isum1 + isum2);
  }

  *s = sum;
}
#endif

#if QK_K == 256
void ggml_vec_dot_q3_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const block_q3_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  uint32_t aux[3];
  uint32_t utmp[4];

  const uint8x16_t m3b = vdupq_n_u8(0x3);
  const int32x4_t vzero = vdupq_n_s32(0);

  const uint8x16_t m0 = vdupq_n_u8(1);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);
  const int8_t m32 = 32;

  int8x16x4_t q3bytes;

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

    const uint8_t *restrict q3 = x[i].qs;
    const uint8_t *restrict qh = x[i].hmask;
    const int8_t *restrict q8 = y[i].qs;

    uint8x16x2_t qhbits = vld1q_u8_x2(qh);

    uint8x16x4_t q3h;

    int32_t isum = 0;

    // Set up scales
    memcpy(aux, x[i].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    int8_t *scale = (int8_t *)utmp;
    for (int j = 0; j < 16; ++j) scale[j] -= m32;

    for (int j = 0; j < QK_K / 128; ++j) {
      const uint8x16x2_t q3bits = vld1q_u8_x2(q3);
      q3 += 32;
      const int8x16x4_t q8bytes_1 = vld1q_s8_x4(q8);
      q8 += 64;
      const int8x16x4_t q8bytes_2 = vld1q_s8_x4(q8);
      q8 += 64;

      q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
      q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
      q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
      q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

      q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];
      scale += 4;

      q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
      q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
      q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
      q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

      q3bytes.val[0] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
      isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];
      scale += 4;

      if (j == 0) {
        qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
        qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
      }
    }
    sum += d * isum;
  }

  *s = sum;
}

#else

void ggml_vec_dot_q3_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q3_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  const int32x4_t vzero = vdupq_n_s32(0);

  const uint8x16_t m3b = vdupq_n_u8(0x3);
  const uint8x16_t mh = vdupq_n_u8(4);

  int8x16x4_t q3bytes;

  uint16_t aux16[2];
  int8_t *scales = (int8_t *)aux16;

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    uint8x16x4_t q3h;

    const uint8x8_t hbits = vld1_u8(x[i].hmask);
    const uint8x16_t q3bits = vld1q_u8(x[i].qs);
    const int8x16x4_t q8bytes = vld1q_s8_x4(y[i].qs);

    const uint16_t a = *(const uint16_t *)x[i].scales;
    aux16[0] = a & 0x0f0f;
    aux16[1] = (a >> 4) & 0x0f0f;

    for (int j = 0; j < 4; ++j) scales[j] -= 8;

    int32_t isum = -4 * (scales[0] * y[i].bsums[0] + scales[2] * y[i].bsums[1] + scales[1] * y[i].bsums[2] +
                         scales[3] * y[i].bsums[3]);

    const float d = y[i].d * (float)x[i].d;

    const uint8x16_t htmp = vcombine_u8(hbits, vshr_n_u8(hbits, 1));
    q3h.val[0] = vandq_u8(mh, vshlq_n_u8(htmp, 2));
    q3h.val[1] = vandq_u8(mh, htmp);
    q3h.val[2] = vandq_u8(mh, vshrq_n_u8(htmp, 2));
    q3h.val[3] = vandq_u8(mh, vshrq_n_u8(htmp, 4));

    q3bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q3bits, m3b), q3h.val[0]));
    q3bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 2), m3b), q3h.val[1]));
    q3bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 4), m3b), q3h.val[2]));
    q3bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q3bits, 6), q3h.val[3]));

    isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes.val[0])) * scales[0];
    isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes.val[1])) * scales[2];
    isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes.val[2])) * scales[1];
    isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes.val[3])) * scales[3];

    sum += d * isum;
  }

  *s = sum;
}
#endif

#if QK_K == 256
void ggml_vec_dot_q4_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q4_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);

  int8x16x2_t q4bytes;
  int8x16x2_t q8bytes;

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
    const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    memcpy(utmp, x[i].scales, 12);

    const uint32x2_t mins8 = {utmp[1] & kmask1, ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)};
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
    const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                     vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    sumf -= dmin * vaddvq_s32(prod);

    const uint8_t *scales = (const uint8_t *)utmp;

    const uint8_t *restrict q4 = x[i].qs;
    const int8_t *restrict q8 = y[i].qs;

    // int32x4_t isum = mzero;

    int32_t sumi1 = 0;
    int32_t sumi2 = 0;

    for (int j = 0; j < QK_K / 64; ++j) {
      const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
      q4 += 32;

      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
      q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

      const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
      sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];

      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
      q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

      const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

      sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
    }

    sumf += d * (sumi1 + sumi2);
  }

  *s = sumf;
}
#else
void ggml_vec_dot_q4_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q4_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  const uint8x16_t m4b = vdupq_n_u8(0xf);

  const int32x4_t mzero = vdupq_n_s32(0);

  float sumf = 0;

  int8x16x2_t q4bytes;
  int8x16x4_t q8bytes;

  float sum_mins = 0.f;

  uint16_t aux16[2];
  const uint8_t *restrict scales = (const uint8_t *)aux16;

  for (int i = 0; i < nb; ++i) {
    const uint8_t *restrict q4 = x[i].qs;
    const int8_t *restrict q8 = y[i].qs;

    const uint16_t *restrict a = (const uint16_t *)x[i].scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
    sum_mins += y[i].d * (float)x[i].d[1] * summi;

    const float d = y[i].d * (float)x[i].d[0];

    const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

    q8bytes = vld1q_s8_x4(q8);
    q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
    q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

    const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
    const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

    q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
    q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

    const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
    const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

    sumf += d * (sumi1 + sumi2);
  }

  *s = sumf - sum_mins;
}
#endif

#if QK_K == 256
void ggml_vec_dot_q5_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q5_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  const uint8x16_t mone = vdupq_n_u8(1);
  const uint8x16_t mtwo = vdupq_n_u8(2);

  int8x16x4_t q5bytes;

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
    const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const uint8x8_t mins8 = vld1_u8((const uint8_t *)utmp + 8);
    const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
    const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                     vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    int32_t sumi_mins = vaddvq_s32(prod);

    const uint8_t *scales = (const uint8_t *)utmp;

    const uint8_t *restrict q5 = x[i].qs;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict q8 = y[i].qs;

    uint8x16x2_t qhbits = vld1q_u8_x2(qh);

    uint8x16x4_t q5h;

    int32_t sumi = 0;

    for (int j = 0; j < QK_K / 64; ++j) {
      const uint8x16x2_t q5bits = vld1q_u8_x2(q5);
      q5 += 32;
      const int8x16x4_t q8bytes = vld1q_s8_x4(q8);
      q8 += 64;

      q5h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q5h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      q5h.val[2] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[0]), 3);
      q5h.val[3] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[1]), 3);
      qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 2);
      qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 2);

      q5bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0], m4b), q5h.val[0]));
      q5bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1], m4b), q5h.val[1]));
      q5bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0], 4), q5h.val[2]));
      q5bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1], 4), q5h.val[3]));

      sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]), q5bytes.val[1], q8bytes.val[1])) *
              *scales++;
      sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]), q5bytes.val[3], q8bytes.val[3])) *
              *scales++;
    }

    sumf += d * sumi - dmin * sumi_mins;
  }

  *s = sumf;
}

#else

void ggml_vec_dot_q5_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q5_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  const uint8x16_t mh = vdupq_n_u8(16);

  int8x16x4_t q5bytes;
  uint8x16x4_t q5h;

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * (float)x[i].d;
    const int8_t *sc = x[i].scales;

    const uint8_t *restrict q5 = x[i].qs;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict q8 = y[i].qs;

    const uint8x8_t qhbits = vld1_u8(qh);

    const uint8x16x2_t q5bits = vld1q_u8_x2(q5);
    const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

    const uint8x16_t htmp = vcombine_u8(qhbits, vshr_n_u8(qhbits, 1));
    q5h.val[0] = vbicq_u8(mh, vshlq_n_u8(htmp, 4));
    q5h.val[1] = vbicq_u8(mh, vshlq_n_u8(htmp, 2));
    q5h.val[2] = vbicq_u8(mh, htmp);
    q5h.val[3] = vbicq_u8(mh, vshrq_n_u8(htmp, 2));

    q5bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[0], m4b)), vreinterpretq_s8_u8(q5h.val[0]));
    q5bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[1], m4b)), vreinterpretq_s8_u8(q5h.val[1]));
    q5bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[0], 4)), vreinterpretq_s8_u8(q5h.val[2]));
    q5bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[1], 4)), vreinterpretq_s8_u8(q5h.val[3]));

    int32_t sumi1 = sc[0] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]));
    int32_t sumi2 = sc[1] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[1], q8bytes.val[1]));
    int32_t sumi3 = sc[2] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]));
    int32_t sumi4 = sc[3] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[3], q8bytes.val[3]));

    sumf += d * (sumi1 + sumi2 + sumi3 + sumi4);
  }

  *s = sumf;
}
#endif

#if QK_K == 256
void ggml_vec_dot_q6_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int32x4_t vzero = vdupq_n_s32(0);
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  int8x16x4_t q6bytes;
  uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = ggml_fp16_to_fp32(x[i].d);

    const uint8_t *restrict q6 = x[i].ql;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict q8 = y[i].qs;

    const int8_t *restrict scale = x[i].scales;

    const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
    const int8x16_t scales = vld1q_s8(scale);
    const int16x8x2_t q6scales = {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))};

    const int32x4_t prod =
        vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                            vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                  vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                            vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
    int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      uint8x16x2_t qhbits = vld1q_u8_x2(qh);
      qh += 32;
      uint8x16x4_t q6bits = vld1q_u8_x4(q6);
      q6 += 64;
      int8x16x4_t q8bytes = vld1q_s8_x4(q8);
      q8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;

      q8bytes = vld1q_s8_x4(q8);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
              vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;
    }
    // sum += isum * d_all * y[i].d;
    sum += d_all * y[i].d * (isum - 32 * isum_mins);
  }
  *s = sum;
}

#else

void ggml_vec_dot_q6_K_q8_K(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K *restrict x = vx;
  const block_q8_K *restrict y = vy;

  const int nb = n / QK_K;

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int32x4_t vzero = vdupq_n_s32(0);
  const int8x16_t m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  int8x16x4_t q6bytes;
  uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = (float)x[i].d;

    const uint8_t *restrict q6 = x[i].ql;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict q8 = y[i].qs;

    const int8_t *restrict scale = x[i].scales;

    int32_t isum = 0;

    uint8x16_t qhbits = vld1q_u8(qh);
    uint8x16x2_t q6bits = vld1q_u8_x2(q6);
    int8x16x4_t q8bytes = vld1q_s8_x4(q8);

    q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
    uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
    q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 4);
    q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 6);
    q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

    q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
    q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
    q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
    q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

    isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
            vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];

    sum += isum * d_all * y[i].d;
  }
  *s = sum;
}

#endif
