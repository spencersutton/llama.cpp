#include "ggml.h"

#include <alloca.h>
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "k_quants.h"

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL 2

//
// logging
//

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN 4
#else
#define GGML_MEM_ALIGN 16
#endif

//
// logging
//

#define GGML_PRINT_DEBUG(...)
#define GGML_PRINT_DEBUG_5(...)

//
// end of logging block
//

inline static void *ggml_aligned_malloc(size_t size) {
  void *aligned_memory = NULL;
#ifdef GGML_USE_METAL
  int result = posix_memalign(&aligned_memory, getpagesize(), size);
#else
  int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
  if (result != 0) {
    // Handle allocation failure
    const char *error_desc = "unknown allocation error";
    switch (result) {
      case EINVAL:
        error_desc = "invalid alignment value";
        break;
      case ENOMEM:
        error_desc = "insufficient memory";
        break;
    }
    printf("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
    return NULL;
  }
  return aligned_memory;
}

#define UNUSED GGML_UNUSED

//
// tensor access macros
//

#define GGML_TENSOR_UNARY_OP_LOCALS           \
  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne); \
  GGML_TENSOR_LOCALS(size_t, nb0, src0, nb);  \
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);   \
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

#define GGML_TENSOR_BINARY_OP_LOCALS          \
  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne); \
  GGML_TENSOR_LOCALS(size_t, nb0, src0, nb);  \
  GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne); \
  GGML_TENSOR_LOCALS(size_t, nb1, src1, nb);  \
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);   \
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

#include <Accelerate/Accelerate.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// floating point type used to accumulate sums
typedef double ggml_float;

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define GGML_COMPUTE_FP16_TO_FP32(x) ((float)(x))
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define GGML_FP16_TO_FP32(x) ((float)(x))
#define GGML_FP32_TO_FP16(x) (x)

#else

#include <immintrin.h>

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#endif  // __ARM_NEON

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static ggml_fp16_t table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
static ggml_fp16_t table_gelu_quick_f16[1 << 16];

// precomputed silu table for f16 (128 KB)
static ggml_fp16_t table_silu_f16[1 << 16];

// precomputed exp table for f16 (128 KB)
static ggml_fp16_t table_exp_f16[1 << 16];

// precomputed f32 table for f16 (256 KB)
static float table_f32_f16[1 << 16];

#if defined(__ARM_NEON) || defined(__wasm_simd128__)
#define B1(c, s, n) 0x##n##c, 0x##n##s
#define B2(c, s, n) B1(c, s, n##c), B1(c, s, n##s)
#define B3(c, s, n) B2(c, s, n##c), B2(c, s, n##s)
#define B4(c, s, n) B3(c, s, n##c), B3(c, s, n##s)
#define B5(c, s, n) B4(c, s, n##c), B4(c, s, n##s)
#define B6(c, s, n) B5(c, s, n##c), B5(c, s, n##s)
#define B7(c, s, n) B6(c, s, n##c), B6(c, s, n##s)
#define B8(c, s) B7(c, s, c), B7(c, s, s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = {B8(00, 10)};  // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = {B8(10, 00)};  // (!b) << 4
#endif

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into ggml_lookup_fp16_to_fp32,
// so we define GGML_FP16_TO_FP32 and GGML_FP32_TO_FP16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(GGML_FP16_TO_FP32) || !defined(GGML_FP32_TO_FP16)

inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
  uint16_t s;
  memcpy(&s, &f, sizeof(uint16_t));
  return table_f32_f16[s];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#endif

// note: do not use these inside ggml.c
// these are meant to be used via the ggml.h API
float ggml_fp16_to_fp32(ggml_fp16_t x) { return (float)GGML_FP16_TO_FP32(x); }

ggml_fp16_t ggml_fp32_to_fp16(float x) { return x; }

void ggml_fp16_to_fp32_row(const ggml_fp16_t *x, float *y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = GGML_FP16_TO_FP32(x[i]);
  }
}

void ggml_fp32_to_fp16_row(const float *x, ggml_fp16_t *y, int n) {
  int i = 0;
  for (; i < n; i++) {
    y[i] = x[i];
  }
}

//
// timing
//

void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t ggml_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int64_t ggml_cycles(void) { return clock(); }

int64_t ggml_cycles_per_ms(void) { return CLOCKS_PER_SEC / 1000; }

#define ggml_perf_time_ms() 0
#define ggml_perf_time_us() 0
#define ggml_perf_cycles() 0
#define ggml_perf_cycles_per_ms() 0

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#define CACHE_LINE_SIZE 64
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE / sizeof(float);

//
// quantization
//

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__ARM_NEON)

#endif

#define QK4_0 32
typedef struct {
  ggml_fp16_t d;          // delta
  uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
  ggml_fp16_t d;          // delta
  ggml_fp16_t m;          // min
  uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
  ggml_fp16_t d;          // delta
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2,
              "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
  ggml_fp16_t d;          // delta
  ggml_fp16_t m;          // min
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2,
              "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
  ggml_fp16_t d;     // delta
  int8_t qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
  float d;           // delta
  float s;           // d * sum(qs[i])
  int8_t qs[QK8_1];  // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

// reference implementation for deterministic creation of model files
static void quantize_row_q4_0_reference(const float *restrict x, block_q4_0 *restrict y, int k) {
  static const int qk = QK4_0;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_0(const float *restrict x, void *restrict y, int k) {
  quantize_row_q4_0_reference(x, y, k);
}

static void quantize_row_q4_1_reference(const float *restrict x, block_q4_1 *restrict y, int k) {
  const int qk = QK4_1;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];

      if (v < min) {
        min = v;
      }
      if (v > max) {
        max = v;
      }
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;
    y[i].m = min;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_1(const float *restrict x, void *restrict y, int k) {
  quantize_row_q4_1_reference(x, y, k);
}

static void quantize_row_q5_0_reference(const float *restrict x, block_q5_0 *restrict y, int k) {
  static const int qk = QK5_0;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
      const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(qh));
  }
}

static void quantize_row_q5_0(const float *restrict x, void *restrict y, int k) {
  quantize_row_q5_0_reference(x, y, k);
}

static void quantize_row_q5_1_reference(const float *restrict x, block_q5_1 *restrict y, int k) {
  const int qk = QK5_1;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];

      if (v < min) {
        min = v;
      }
      if (v > max) {
        max = v;
      }
    }

    const float d = (max - min) / ((1 << 5) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;
    y[i].m = min;

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
      const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
  }
}

static void quantize_row_q5_1(const float *restrict x, void *restrict y, int k) {
  quantize_row_q5_1_reference(x, y, k);
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_0_reference(const float *restrict x, block_q8_0 *restrict y, int k) {
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

static void quantize_row_q8_0(const float *restrict x, void *restrict vy, int k) {
  const int nb = k / QK8_0;

  block_q8_0 *restrict y = vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float32x4_t srcv[8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int j = 0; j < 8; j++) {
      srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
    }
    for (int j = 0; j < 8; j++) {
      asrcv[j] = vabsq_f32(srcv[j]);
    }

    for (int j = 0; j < 4; j++) {
      amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
    }
    for (int j = 0; j < 2; j++) {
      amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
    }
    for (int j = 0; j < 1; j++) {
      amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);
    }

    const float amax = vmaxvq_f32(amaxv[0]);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    for (int j = 0; j < 8; j++) {
      const float32x4_t v = vmulq_n_f32(srcv[j], id);
      const int32x4_t vi = vcvtnq_s32_f32(v);

      y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);
    }
  }
#else
  // scalar
  quantize_row_q8_0_reference(x, y, k);
#endif
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_1_reference(const float *restrict x, block_q8_1 *restrict y, int k) {
  const int nb = k / QK8_1;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_1; j++) {
      const float v = x[i * QK8_1 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    int sum = 0;

    for (int j = 0; j < QK8_1 / 2; ++j) {
      const float v0 = x[i * QK8_1 + j] * id;
      const float v1 = x[i * QK8_1 + QK8_1 / 2 + j] * id;

      y[i].qs[j] = roundf(v0);
      y[i].qs[QK8_1 / 2 + j] = roundf(v1);

      sum += y[i].qs[j];
      sum += y[i].qs[QK8_1 / 2 + j];
    }

    y[i].s = sum * d;
  }
}

static void quantize_row_q8_1(const float *restrict x, void *restrict vy, int k) {
  const int nb = k / QK8_1;

  block_q8_1 *restrict y = vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float32x4_t srcv[8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int j = 0; j < 8; j++) {
      srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
    }
    for (int j = 0; j < 8; j++) {
      asrcv[j] = vabsq_f32(srcv[j]);
    }

    for (int j = 0; j < 4; j++) {
      amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
    }
    for (int j = 0; j < 2; j++) {
      amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
    }
    for (int j = 0; j < 1; j++) {
      amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);
    }

    const float amax = vmaxvq_f32(amaxv[0]);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    int32x4_t accv = vdupq_n_s32(0);

    for (int j = 0; j < 8; j++) {
      const float32x4_t v = vmulq_n_f32(srcv[j], id);
      const int32x4_t vi = vcvtnq_s32_f32(v);

      y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);

      accv = vaddq_s32(accv, vi);
    }

    y[i].s = d * vaddvq_s32(accv);
  }
#else
  // scalar
  quantize_row_q8_1_reference(x, y, k);
#endif
}

static void dequantize_row_q4_0(const block_q4_0 *restrict x, float *restrict y, int k) {
  static const int qk = QK4_0;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F) - 8;
      const int x1 = (x[i].qs[j] >> 4) - 8;

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

static void dequantize_row_q4_1(const block_q4_1 *restrict x, float *restrict y, int k) {
  static const int qk = QK4_1;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    const float m = GGML_FP16_TO_FP32(x[i].m);

    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F);
      const int x1 = (x[i].qs[j] >> 4);

      y[i * qk + j + 0] = x0 * d + m;
      y[i * qk + j + qk / 2] = x1 * d + m;
    }
  }
}

static void dequantize_row_q5_0(const block_q5_0 *restrict x, float *restrict y, int k) {
  static const int qk = QK5_0;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

static void dequantize_row_q5_1(const block_q5_1 *restrict x, float *restrict y, int k) {
  static const int qk = QK5_1;

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    const float m = GGML_FP16_TO_FP32(x[i].m);

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
      const int x1 = (x[i].qs[j] >> 4) | xh_1;

      y[i * qk + j + 0] = x0 * d + m;
      y[i * qk + j + qk / 2] = x1 * d + m;
    }
  }
}

static void dequantize_row_q8_0(const void *restrict vx, float *restrict y, int k) {
  static const int qk = QK8_0;

  const int nb = k / qk;

  const block_q8_0 *restrict x = vx;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}

static void ggml_vec_dot_f32(int n, float *restrict s, const float *restrict x, const float *restrict y);
static void ggml_vec_dot_f16(int n, float *restrict s, ggml_fp16_t *restrict x, ggml_fp16_t *restrict y);
static void ggml_vec_dot_q4_0_q8_0(int n, float *restrict s, const void *restrict vx, const void *restrict vy);
static void ggml_vec_dot_q4_1_q8_1(int n, float *restrict s, const void *restrict vx, const void *restrict vy);
static void ggml_vec_dot_q5_0_q8_0(int n, float *restrict s, const void *restrict vx, const void *restrict vy);
static void ggml_vec_dot_q5_1_q8_1(int n, float *restrict s, const void *restrict vx, const void *restrict vy);
static void ggml_vec_dot_q8_0_q8_0(int n, float *restrict s, const void *restrict vx, const void *restrict vy);

static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] =
        {
            .vec_dot = (ggml_vec_dot_t)ggml_vec_dot_f32,
            .vec_dot_type = GGML_TYPE_F32,
        },
    [GGML_TYPE_F16] =
        {
            .to_float = (ggml_to_float_t)ggml_fp16_to_fp32_row,
            .from_float = (ggml_from_float_t)ggml_fp32_to_fp16_row,
            .from_float_reference = (ggml_from_float_t)ggml_fp32_to_fp16_row,
            .vec_dot = (ggml_vec_dot_t)ggml_vec_dot_f16,
            .vec_dot_type = GGML_TYPE_F16,
        },
    [GGML_TYPE_Q4_0] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q4_0,
            .from_float = quantize_row_q4_0,
            .from_float_reference = (ggml_from_float_t)quantize_row_q4_0_reference,
            .vec_dot = ggml_vec_dot_q4_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
        },
    [GGML_TYPE_Q4_1] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q4_1,
            .from_float = quantize_row_q4_1,
            .from_float_reference = (ggml_from_float_t)quantize_row_q4_1_reference,
            .vec_dot = ggml_vec_dot_q4_1_q8_1,
            .vec_dot_type = GGML_TYPE_Q8_1,
        },
    [GGML_TYPE_Q5_0] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q5_0,
            .from_float = quantize_row_q5_0,
            .from_float_reference = (ggml_from_float_t)quantize_row_q5_0_reference,
            .vec_dot = ggml_vec_dot_q5_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
        },
    [GGML_TYPE_Q5_1] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q5_1,
            .from_float = quantize_row_q5_1,
            .from_float_reference = (ggml_from_float_t)quantize_row_q5_1_reference,
            .vec_dot = ggml_vec_dot_q5_1_q8_1,
            .vec_dot_type = GGML_TYPE_Q8_1,
        },
    [GGML_TYPE_Q8_0] =
        {
            .to_float = dequantize_row_q8_0,
            .from_float = quantize_row_q8_0,
            .from_float_reference = (ggml_from_float_t)quantize_row_q8_0_reference,
            .vec_dot = ggml_vec_dot_q8_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
        },
    [GGML_TYPE_Q8_1] =
        {
            .from_float = quantize_row_q8_1,
            .from_float_reference = (ggml_from_float_t)quantize_row_q8_1_reference,
            .vec_dot_type = GGML_TYPE_Q8_1,
        },
    [GGML_TYPE_Q2_K] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q2_K,
            .from_float = quantize_row_q2_K,
            .from_float_reference = (ggml_from_float_t)quantize_row_q2_K_reference,
            .vec_dot = ggml_vec_dot_q2_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
        },
    [GGML_TYPE_Q3_K] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q3_K,
            .from_float = quantize_row_q3_K,
            .from_float_reference = (ggml_from_float_t)quantize_row_q3_K_reference,
            .vec_dot = ggml_vec_dot_q3_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
        },
    [GGML_TYPE_Q4_K] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q4_K,
            .from_float = quantize_row_q4_K,
            .from_float_reference = (ggml_from_float_t)quantize_row_q4_K_reference,
            .vec_dot = ggml_vec_dot_q4_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
        },
    [GGML_TYPE_Q5_K] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q5_K,
            .from_float = quantize_row_q5_K,
            .from_float_reference = (ggml_from_float_t)quantize_row_q5_K_reference,
            .vec_dot = ggml_vec_dot_q5_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
        },
    [GGML_TYPE_Q6_K] =
        {
            .to_float = (ggml_to_float_t)dequantize_row_q6_K,
            .from_float = quantize_row_q6_K,
            .from_float_reference = (ggml_from_float_t)quantize_row_q6_K_reference,
            .vec_dot = ggml_vec_dot_q6_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
        },
    [GGML_TYPE_Q8_K] = {
        .from_float = quantize_row_q8_K,
    }};

// For internal test use
ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type i) { return type_traits[i]; }

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define GGML_SIMD

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR 4

#define GGML_F32x4 float32x4_t
#define GGML_F32x4_ZERO vdupq_n_f32(0.0f)
#define GGML_F32x4_SET1(x) vdupq_n_f32(x)
#define GGML_F32x4_LOAD vld1q_f32
#define GGML_F32x4_STORE vst1q_f32
#define GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define GGML_F32x4_ADD vaddq_f32
#define GGML_F32x4_MUL vmulq_f32
#define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define GGML_F32x4_REDUCE(res, x)                  \
  {                                                \
    int offset = GGML_F32_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
      (x)[i] = vaddq_f32((x)[i], (x)[offset + i]); \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
      (x)[i] = vaddq_f32((x)[i], (x)[offset + i]); \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
      (x)[i] = vaddq_f32((x)[i], (x)[offset + i]); \
    }                                              \
    (res) = GGML_F32x4_REDUCE_ONE((x)[0]);         \
  }

#define GGML_F32_VEC GGML_F32x4
#define GGML_F32_VEC_ZERO GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1 GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE GGML_F32x4_STORE
#define GGML_F32_VEC_FMA GGML_F32x4_FMA
#define GGML_F32_VEC_ADD GGML_F32x4_ADD
#define GGML_F32_VEC_MUL GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define GGML_F16_STEP 32
#define GGML_F16_EPR 8

#define GGML_F16x8 float16x8_t
#define GGML_F16x8_ZERO vdupq_n_f16(0.0f)
#define GGML_F16x8_SET1(x) vdupq_n_f16(x)
#define GGML_F16x8_LOAD vld1q_f16
#define GGML_F16x8_STORE vst1q_f16
#define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
#define GGML_F16x8_ADD vaddq_f16
#define GGML_F16x8_MUL vmulq_f16
#define GGML_F16x8_REDUCE(res, x)                               \
  {                                                             \
    int offset = GGML_F16_ARR >> 1;                             \
    for (int i = 0; i < offset; ++i) {                          \
      (x)[i] = vaddq_f16((x)[i], (x)[offset + i]);              \
    }                                                           \
    offset >>= 1;                                               \
    for (int i = 0; i < offset; ++i) {                          \
      (x)[i] = vaddq_f16((x)[i], (x)[offset + i]);              \
    }                                                           \
    offset >>= 1;                                               \
    for (int i = 0; i < offset; ++i) {                          \
      (x)[i] = vaddq_f16((x)[i], (x)[offset + i]);              \
    }                                                           \
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16((x)[0]));  \
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16((x)[0])); \
    (res) = (ggml_float)vaddvq_f32(vaddq_f32(t0, t1));          \
  }

#define GGML_F16_VEC GGML_F16x8
#define GGML_F16_VEC_ZERO GGML_F16x8_ZERO
#define GGML_F16_VEC_SET1 GGML_F16x8_SET1
#define GGML_F16_VEC_LOAD(p, i) GGML_F16x8_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE(p, (r)[i])
#define GGML_F16_VEC_FMA GGML_F16x8_FMA
#define GGML_F16_VEC_ADD GGML_F16x8_ADD
#define GGML_F16_VEC_MUL GGML_F16x8_MUL
#define GGML_F16_VEC_REDUCE GGML_F16x8_REDUCE
#else
// if FP16 vector arithmetic is not supported, we use FP32 instead
// and take advantage of the vcvt_ functions to convert to/from FP16

#define GGML_F16_STEP 16
#define GGML_F16_EPR 4

#define GGML_F32Cx4 float32x4_t
#define GGML_F32Cx4_ZERO vdupq_n_f32(0.0f)
#define GGML_F32Cx4_SET1(x) vdupq_n_f32(x)
#define GGML_F32Cx4_LOAD(x) vcvt_f32_f16(vld1_f16(x))
#define GGML_F32Cx4_STORE(x, y) vst1_f16(x, vcvt_f16_f32(y))
#define GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define GGML_F32Cx4_ADD vaddq_f32
#define GGML_F32Cx4_MUL vmulq_f32
#define GGML_F32Cx4_REDUCE GGML_F32x4_REDUCE

#define GGML_F16_VEC GGML_F32Cx4
#define GGML_F16_VEC_ZERO GGML_F32Cx4_ZERO
#define GGML_F16_VEC_SET1 GGML_F32Cx4_SET1
#define GGML_F16_VEC_LOAD(p, i) GGML_F32Cx4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE(p, r[i])
#define GGML_F16_VEC_FMA GGML_F32Cx4_FMA
#define GGML_F16_VEC_ADD GGML_F32Cx4_ADD
#define GGML_F16_VEC_MUL GGML_F32Cx4_MUL
#define GGML_F16_VEC_REDUCE GGML_F32Cx4_REDUCE
#endif

#endif

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#ifdef GGML_SIMD
#define GGML_F32_ARR (GGML_F32_STEP / GGML_F32_EPR)
#define GGML_F16_ARR (GGML_F16_STEP / GGML_F16_EPR)
#endif

//
// fundamental operations
//

inline static void ggml_vec_set_i8(const int n, int8_t *x, const int8_t v) {
  for (int i = 0; i < n; ++i) {
    x[i] = v;
  }
}

inline static void ggml_vec_set_i16(const int n, int16_t *x, const int16_t v) {
  for (int i = 0; i < n; ++i) {
    x[i] = v;
  }
}

inline static void ggml_vec_set_i32(const int n, int32_t *x, const int32_t v) {
  for (int i = 0; i < n; ++i) {
    x[i] = v;
  }
}

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t *x, const int32_t v) {
  for (int i = 0; i < n; ++i) {
    x[i] = v;
  }
}

inline static void ggml_vec_add_f32(const int n, float *z, const float *x, const float *y) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}
inline static void ggml_vec_add1_f32(const int n, float *z, const float *x, const float v) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + v;
  }
}
inline static void ggml_vec_acc_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] += x[i];
  }
}
inline static void ggml_vec_acc1_f32(const int n, float *y, const float v) {
  for (int i = 0; i < n; ++i) {
    y[i] += v;
  }
}
inline static void ggml_vec_sub_f32(const int n, float *z, const float *x, const float *y) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
}
inline static void ggml_vec_set_f32(const int n, float *x, const float v) {
  for (int i = 0; i < n; ++i) {
    x[i] = v;
  }
}
inline static void ggml_vec_cpy_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}
inline static void ggml_vec_neg_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = -x[i];
  }
}
inline static void ggml_vec_mul_f32(const int n, float *z, const float *x, const float *y) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}
inline static void ggml_vec_div_f32(const int n, float *z, const float *x, const float *y) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] / y[i];
  }
}

static void ggml_vec_dot_f32(const int n, float *restrict s, const float *restrict x, const float *restrict y) {
#ifdef GGML_SIMD
  float sumf = 0.0f;
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC sum[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};

  GGML_F32_VEC ax[GGML_F32_ARR];
  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);

      sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  GGML_F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += x[i] * y[i];
  }
#else
  // scalar
  ggml_float sumf = 0.0;
  for (int i = 0; i < n; ++i) {
    sumf += (ggml_float)(x[i] * y[i]);
  }
#endif

  *s = sumf;
}

static void ggml_vec_dot_f16(const int n, float *restrict s, ggml_fp16_t *restrict x, ggml_fp16_t *restrict y) {
  ggml_float sumf = 0.0;

#if defined(GGML_SIMD)
  const int np = (n & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC sum[GGML_F16_ARR] = {GGML_F16_VEC_ZERO};

  GGML_F16_VEC ax[GGML_F16_ARR];
  GGML_F16_VEC ay[GGML_F16_ARR];

  for (int i = 0; i < np; i += GGML_F16_STEP) {
    for (int j = 0; j < GGML_F16_ARR; j++) {
      ax[j] = GGML_F16_VEC_LOAD(x + i + j * GGML_F16_EPR, j);
      ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);

      sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  GGML_F16_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]));
  }
#else
  for (int i = 0; i < n; ++i) {
    sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]));
  }
#endif

  *s = sumf;
}

static void ggml_vec_dot_q4_0_q8_0(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  const block_q4_0 *restrict x = vx;
  const block_q8_0 *restrict y = vy;

#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; i += 2) {
    const block_q4_0 *restrict x0 = &x[i + 0];
    const block_q4_0 *restrict x1 = &x[i + 1];
    const block_q8_0 *restrict y0 = &y[i + 0];
    const block_q8_0 *restrict y1 = &y[i + 1];

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    const uint8x16_t v0_0 = vld1q_u8(x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(x1->qs);

    // 4-bit -> 8-bit
    const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    // sub 8
    const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
    const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
    const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
    const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

    // load y
    const int8x16_t v1_0l = vld1q_s8(y0->qs);
    const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(y1->qs);
    const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    // dot product into int32x4_t
    const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
    const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
#else
    const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0l));
    const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
    const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0h));
    const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

    const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1l));
    const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
    const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1h));
    const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

    const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
    const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
    const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
    const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F) - 8;
      const int v1 = (x[i].qs[j] >> 4) - 8;

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += sumi * GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d);
  }

  *s = sumf;
#endif
}

static void ggml_vec_dot_q4_1_q8_1(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  const block_q4_1 *restrict x = vx;
  const block_q8_1 *restrict y = vy;

  // TODO: add WASM SIMD
#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  float summs = 0;

  for (int i = 0; i < nb; i += 2) {
    const block_q4_1 *restrict x0 = &x[i + 0];
    const block_q4_1 *restrict x1 = &x[i + 1];
    const block_q8_1 *restrict y0 = &y[i + 0];
    const block_q8_1 *restrict y1 = &y[i + 1];

    summs += GGML_FP16_TO_FP32(x0->m) * y0->s + GGML_FP16_TO_FP32(x1->m) * y1->s;

    const uint8x16_t m4b = vdupq_n_u8(0x0F);

    const uint8x16_t v0_0 = vld1q_u8(x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(x1->qs);

    // 4-bit -> 8-bit
    const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    // load y
    const int8x16_t v1_0l = vld1q_s8(y0->qs);
    const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(y1->qs);
    const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    // dot product into int32x4_t
    const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
    const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d) * y0->d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d) * y1->d);
#else
    const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0l), vget_low_s8(v1_0l));
    const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0l), vget_high_s8(v1_0l));
    const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0h), vget_low_s8(v1_0h));
    const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0h), vget_high_s8(v1_0h));

    const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1l), vget_low_s8(v1_1l));
    const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1l), vget_high_s8(v1_1l));
    const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1h), vget_low_s8(v1_1h));
    const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1h), vget_high_s8(v1_1h));

    const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
    const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
    const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
    const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d) * y0->d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d) * y1->d);
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F);
      const int v1 = (x[i].qs[j] >> 4);

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += (GGML_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + GGML_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ggml_vec_dot_q5_0_q8_0(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  const block_q5_0 *restrict x = vx;
  const block_q8_0 *restrict y = vy;

#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  uint32_t qh0;
  uint32_t qh1;

  uint64_t tmp0[4];
  uint64_t tmp1[4];

  for (int i = 0; i < nb; i += 2) {
    const block_q5_0 *restrict x0 = &x[i];
    const block_q5_0 *restrict x1 = &x[i + 1];
    const block_q8_0 *restrict y0 = &y[i];
    const block_q8_0 *restrict y1 = &y[i + 1];

    const uint8x16_t m4b = vdupq_n_u8(0x0F);

    // extract the 5th bit via lookup table ((!b) << 4)
    memcpy(&qh0, x0->qh, sizeof(qh0));
    memcpy(&qh1, x1->qh, sizeof(qh1));

    tmp0[0] = table_b2b_1[(qh0 >> 0) & 0xFF];
    tmp0[1] = table_b2b_1[(qh0 >> 8) & 0xFF];
    tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
    tmp0[3] = table_b2b_1[(qh0 >> 24)];

    tmp1[0] = table_b2b_1[(qh1 >> 0) & 0xFF];
    tmp1[1] = table_b2b_1[(qh1 >> 8) & 0xFF];
    tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
    tmp1[3] = table_b2b_1[(qh1 >> 24)];

    const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
    const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
    const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
    const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

    const uint8x16_t v0_0 = vld1q_u8(x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(x1->qs);

    // 4-bit -> 8-bit
    int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
    const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
    const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
    const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
    const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

    // load y
    const int8x16_t v1_0l = vld1q_s8(y0->qs);
    const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(y1->qs);
    const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    sumv0 = vmlaq_n_f32(
        sumv0,
        vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l), vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))),
        GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(
        sumv1,
        vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l), vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))),
        GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
#else
    const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0lf), vget_low_s8(v1_0l));
    const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lf), vget_high_s8(v1_0l));
    const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hf), vget_low_s8(v1_0h));
    const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hf), vget_high_s8(v1_0h));

    const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1lf), vget_low_s8(v1_1l));
    const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lf), vget_high_s8(v1_1l));
    const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hf), vget_low_s8(v1_1h));
    const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hf), vget_high_s8(v1_1h));

    const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
    const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
    const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
    const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
      const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d)) * sumi;
  }

  *s = sumf;
#endif
}

static void ggml_vec_dot_q5_1_q8_1(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  const block_q5_1 *restrict x = vx;
  const block_q8_1 *restrict y = vy;

#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  float summs0 = 0.0f;
  float summs1 = 0.0f;

  uint32_t qh0;
  uint32_t qh1;

  uint64_t tmp0[4];
  uint64_t tmp1[4];

  for (int i = 0; i < nb; i += 2) {
    const block_q5_1 *restrict x0 = &x[i];
    const block_q5_1 *restrict x1 = &x[i + 1];
    const block_q8_1 *restrict y0 = &y[i];
    const block_q8_1 *restrict y1 = &y[i + 1];

    const uint8x16_t m4b = vdupq_n_u8(0x0F);

    summs0 += GGML_FP16_TO_FP32(x0->m) * y0->s;
    summs1 += GGML_FP16_TO_FP32(x1->m) * y1->s;

    // extract the 5th bit via lookup table ((b) << 4)
    memcpy(&qh0, x0->qh, sizeof(qh0));
    memcpy(&qh1, x1->qh, sizeof(qh1));

    tmp0[0] = table_b2b_0[(qh0 >> 0) & 0xFF];
    tmp0[1] = table_b2b_0[(qh0 >> 8) & 0xFF];
    tmp0[2] = table_b2b_0[(qh0 >> 16) & 0xFF];
    tmp0[3] = table_b2b_0[(qh0 >> 24)];

    tmp1[0] = table_b2b_0[(qh1 >> 0) & 0xFF];
    tmp1[1] = table_b2b_0[(qh1 >> 8) & 0xFF];
    tmp1[2] = table_b2b_0[(qh1 >> 16) & 0xFF];
    tmp1[3] = table_b2b_0[(qh1 >> 24)];

    const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
    const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
    const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
    const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

    const uint8x16_t v0_0 = vld1q_u8(x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(x1->qs);

    // 4-bit -> 8-bit
    const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    // add high bit
    const int8x16_t v0_0lf = vorrq_s8(v0_0l, qhl0);
    const int8x16_t v0_0hf = vorrq_s8(v0_0h, qhh0);
    const int8x16_t v0_1lf = vorrq_s8(v0_1l, qhl1);
    const int8x16_t v0_1hf = vorrq_s8(v0_1h, qhh1);

    // load y
    const int8x16_t v1_0l = vld1q_s8(y0->qs);
    const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(y1->qs);
    const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    sumv0 = vmlaq_n_f32(
        sumv0,
        vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l), vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))),
        GGML_FP16_TO_FP32(x0->d) * y0->d);
    sumv1 = vmlaq_n_f32(
        sumv1,
        vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l), vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))),
        GGML_FP16_TO_FP32(x1->d) * y1->d);
#else
    const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0lf), vget_low_s8(v1_0l));
    const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lf), vget_high_s8(v1_0l));
    const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hf), vget_low_s8(v1_0h));
    const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hf), vget_high_s8(v1_0h));

    const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1lf), vget_low_s8(v1_1l));
    const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lf), vget_high_s8(v1_1l));
    const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hf), vget_low_s8(v1_1h));
    const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hf), vget_high_s8(v1_1h));

    const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
    const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
    const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
    const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d) * y0->d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d) * y1->d);
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int32_t x0 = (x[i].qs[j] & 0xF) | xh_0;
      const int32_t x1 = (x[i].qs[j] >> 4) | xh_1;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (GGML_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + GGML_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ggml_vec_dot_q8_0_q8_0(const int n, float *restrict s, const void *restrict vx, const void *restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  const block_q8_0 *restrict x = vx;
  const block_q8_0 *restrict y = vy;

#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  for (int i = 0; i < nb; i += 2) {
    const block_q8_0 *restrict x0 = &x[i + 0];
    const block_q8_0 *restrict x1 = &x[i + 1];
    const block_q8_0 *restrict y0 = &y[i + 0];
    const block_q8_0 *restrict y1 = &y[i + 1];

    const int8x16_t x0_0 = vld1q_s8(x0->qs);
    const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
    const int8x16_t x1_0 = vld1q_s8(x1->qs);
    const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

    // load y
    const int8x16_t y0_0 = vld1q_s8(y0->qs);
    const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
    const int8x16_t y1_0 = vld1q_s8(y1->qs);
    const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    sumv0 = vmlaq_n_f32(
        sumv0, vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
        GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));

    sumv1 = vmlaq_n_f32(
        sumv1, vcvtq_f32_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
        GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));

#else
    const int16x8_t p0_0 = vmull_s8(vget_low_s8(x0_0), vget_low_s8(y0_0));
    const int16x8_t p0_1 = vmull_s8(vget_high_s8(x0_0), vget_high_s8(y0_0));
    const int16x8_t p0_2 = vmull_s8(vget_low_s8(x0_1), vget_low_s8(y0_1));
    const int16x8_t p0_3 = vmull_s8(vget_high_s8(x0_1), vget_high_s8(y0_1));

    const int16x8_t p1_0 = vmull_s8(vget_low_s8(x1_0), vget_low_s8(y1_0));
    const int16x8_t p1_1 = vmull_s8(vget_high_s8(x1_0), vget_high_s8(y1_0));
    const int16x8_t p1_2 = vmull_s8(vget_low_s8(x1_1), vget_low_s8(y1_1));
    const int16x8_t p1_3 = vmull_s8(vget_high_s8(x1_1), vget_high_s8(y1_1));

    const int32x4_t p0 = vaddq_s32(vpaddlq_s16(p0_0), vpaddlq_s16(p0_1));
    const int32x4_t p1 = vaddq_s32(vpaddlq_s16(p0_2), vpaddlq_s16(p0_3));
    const int32x4_t p2 = vaddq_s32(vpaddlq_s16(p1_0), vpaddlq_s16(p1_1));
    const int32x4_t p3 = vaddq_s32(vpaddlq_s16(p1_2), vpaddlq_s16(p1_3));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(p0, p1)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(p2, p3)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) {
      sumi += x[i].qs[j] * y[i].qs[j];
    }

    sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
  }

  *s = sumf;
#endif
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float *restrict s, void *restrict xv,
                                           ggml_fp16_t *restrict y) {
  ggml_float sumf[GGML_VEC_DOT_UNROLL] = {0.0};

  ggml_fp16_t *restrict x[GGML_VEC_DOT_UNROLL];

  for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
    x[i] = (ggml_fp16_t *)((char *)xv + i * xs);
  }

#if defined(GGML_SIMD)
  const int np = (n & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = {{GGML_F16_VEC_ZERO}};

  GGML_F16_VEC ax[GGML_F16_ARR];
  GGML_F16_VEC ay[GGML_F16_ARR];

  for (int i = 0; i < np; i += GGML_F16_STEP) {
    for (int j = 0; j < GGML_F16_ARR; j++) {
      ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);

      for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
        ax[j] = GGML_F16_VEC_LOAD(x[k] + i + j * GGML_F16_EPR, j);

        sum[k][j] = GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
      }
    }
  }

  // reduce sum0..sum3 to sum0
  for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
    GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ggml_float)(GGML_FP16_TO_FP32(x[j][i]) * GGML_FP16_TO_FP32(y[i]));
    }
  }
#else
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ggml_float)(GGML_FP16_TO_FP32(x[j][i]) * GGML_FP16_TO_FP32(y[i]));
    }
  }
#endif

  for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
    s[i] = sumf[i];
  }
}

inline static void ggml_vec_mad_f32(const int n, float *restrict y, const float *restrict x, const float v) {
#if defined(GGML_SIMD)
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ax[GGML_F32_ARR];
  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] += x[i] * v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] += x[i] * v;
  }
#endif
}

// inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *=
// v;          }
inline static void ggml_vec_scale_f32(const int n, float *y, const float v) {
#if defined(GGML_SIMD)
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] *= v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] *= v;
  }
#endif
}

inline static void ggml_vec_norm_f32(const int n, float *s, const float *x) {
  ggml_vec_dot_f32(n, s, x, x);
  *s = sqrtf(*s);
}
inline static void ggml_vec_sqr_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * x[i];
  }
}
inline static void ggml_vec_sqrt_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = sqrtf(x[i]);
  }
}
inline static void ggml_vec_log_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = logf(x[i]);
  }
}
inline static void ggml_vec_abs_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = fabsf(x[i]);
  }
}
inline static void ggml_vec_sgn_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f);
  }
}
inline static void ggml_vec_step_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] > 0.f) ? 1.f : 0.f;
  }
}
inline static void ggml_vec_tanh_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = tanhf(x[i]);
  }
}
inline static void ggml_vec_elu_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] > 0.f) ? x[i] : expf(x[i]) - 1;
  }
}
inline static void ggml_vec_relu_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] > 0.f) ? x[i] : 0.f;
  }
}

static const float GELU_COEF_A = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

inline static float ggml_gelu_f32(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t *y, const ggml_fp16_t *x) {
  const uint16_t *i16 = (const uint16_t *)x;
  for (int i = 0; i < n; ++i) {
    y[i] = table_gelu_f16[i16[i]];
  }
}

inline static void ggml_vec_gelu_f32(const int n, float *y, const float *x) {
  uint16_t t;
  for (int i = 0; i < n; ++i) {
    ggml_fp16_t fp16 = x[i];
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = GGML_FP16_TO_FP32(table_gelu_f16[t]);
  }
}

inline static float ggml_gelu_quick_f32(float x) { return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x))); }

inline static void ggml_vec_gelu_quick_f32(const int n, float *y, const float *x) {
  uint16_t t;
  for (int i = 0; i < n; ++i) {
    ggml_fp16_t fp16 = x[i];
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = GGML_FP16_TO_FP32(table_gelu_quick_f16[t]);
  }
}

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float x) { return x / (1.0f + expf(-x)); }

inline static void ggml_vec_silu_f32(const int n, float *y, const float *x) {
  uint16_t t;
  for (int i = 0; i < n; ++i) {
    ggml_fp16_t fp16 = x[i];
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = GGML_FP16_TO_FP32(table_silu_f16[t]);
  }
}

inline static float ggml_silu_backward_f32(float x, float dy) {
  const float s = 1.0f / (1.0f + expf(-x));
  return dy * s * (1.0f + x * (1.0f - s));
}

inline static void ggml_vec_silu_backward_f32(const int n, float *dx, const float *x, const float *dy) {
  for (int i = 0; i < n; ++i) {
    // we did not use x[i] to compute forward silu but its f16 equivalent
    // take derivative at f16 of x[i]:
    ggml_fp16_t fp16 = x[i];
    float usedx = GGML_FP16_TO_FP32(fp16);
    dx[i] = ggml_silu_backward_f32(usedx, dy[i]);
  }
}

inline static void ggml_vec_sum_f32(const int n, float *s, const float *x) { vDSP_sve(x, 1, s, n); }

inline static void ggml_vec_sum_ggf(const int n, ggml_float *s, const float *x) {
  ggml_float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += (ggml_float)x[i];
  }
  *s = sum;
}

inline static void ggml_vec_max_f32(const int n, float *s, const float *x) { vDSP_maxv(x, 1, s, n); }

inline static void ggml_vec_norm_inv_f32(const int n, float *s, const float *x) {
  ggml_vec_norm_f32(n, s, x);
  *s = 1.f / (*s);
}

inline static void ggml_vec_argmax_f32(const int n, int *s, const float *x) {
  float max = -INFINITY;
  int idx = 0;
  for (int i = 0; i < n; ++i) {
    max = MAX(max, x[i]);
    if (max == x[i]) {
      idx = i;
    }
  }
  *s = idx;
}

//
// data types
//

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = 1,      [GGML_TYPE_F16] = 1,      [GGML_TYPE_Q4_0] = QK4_0, [GGML_TYPE_Q4_1] = QK4_1,
    [GGML_TYPE_Q5_0] = QK5_0, [GGML_TYPE_Q5_1] = QK5_1, [GGML_TYPE_Q8_0] = QK8_0, [GGML_TYPE_Q8_1] = QK8_1,
    [GGML_TYPE_Q2_K] = QK_K,  [GGML_TYPE_Q3_K] = QK_K,  [GGML_TYPE_Q4_K] = QK_K,  [GGML_TYPE_Q5_K] = QK_K,
    [GGML_TYPE_Q6_K] = QK_K,  [GGML_TYPE_Q8_K] = QK_K,  [GGML_TYPE_I8] = 1,       [GGML_TYPE_I16] = 1,
    [GGML_TYPE_I32] = 1,
};
static_assert(GGML_TYPE_COUNT == 19, "GGML_BLCK_SIZE is outdated");

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = sizeof(float),       [GGML_TYPE_F16] = sizeof(ggml_fp16_t), [GGML_TYPE_Q4_0] = sizeof(block_q4_0),
    [GGML_TYPE_Q4_1] = sizeof(block_q4_1), [GGML_TYPE_Q5_0] = sizeof(block_q5_0), [GGML_TYPE_Q5_1] = sizeof(block_q5_1),
    [GGML_TYPE_Q8_0] = sizeof(block_q8_0), [GGML_TYPE_Q8_1] = sizeof(block_q8_1), [GGML_TYPE_Q2_K] = sizeof(block_q2_K),
    [GGML_TYPE_Q3_K] = sizeof(block_q3_K), [GGML_TYPE_Q4_K] = sizeof(block_q4_K), [GGML_TYPE_Q5_K] = sizeof(block_q5_K),
    [GGML_TYPE_Q6_K] = sizeof(block_q6_K), [GGML_TYPE_Q8_K] = sizeof(block_q8_K), [GGML_TYPE_I8] = sizeof(int8_t),
    [GGML_TYPE_I16] = sizeof(int16_t),     [GGML_TYPE_I32] = sizeof(int32_t),
};
static_assert(GGML_TYPE_COUNT == 19, "GGML_TYPE_SIZE is outdated");

static const char *GGML_TYPE_NAME[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = "f32",   [GGML_TYPE_F16] = "f16",   [GGML_TYPE_Q4_0] = "q4_0", [GGML_TYPE_Q4_1] = "q4_1",
    [GGML_TYPE_Q5_0] = "q5_0", [GGML_TYPE_Q5_1] = "q5_1", [GGML_TYPE_Q8_0] = "q8_0", [GGML_TYPE_Q8_1] = "q8_1",
    [GGML_TYPE_Q2_K] = "q2_K", [GGML_TYPE_Q3_K] = "q3_K", [GGML_TYPE_Q4_K] = "q4_K", [GGML_TYPE_Q5_K] = "q5_K",
    [GGML_TYPE_Q6_K] = "q6_K", [GGML_TYPE_Q8_K] = "q8_K", [GGML_TYPE_I8] = "i8",     [GGML_TYPE_I16] = "i16",
    [GGML_TYPE_I32] = "i32",
};
static_assert(GGML_TYPE_COUNT == 19, "GGML_TYPE_NAME is outdated");

static bool GGML_IS_QUANTIZED[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = false, [GGML_TYPE_F16] = false, [GGML_TYPE_Q4_0] = true, [GGML_TYPE_Q4_1] = true,
    [GGML_TYPE_Q5_0] = true, [GGML_TYPE_Q5_1] = true, [GGML_TYPE_Q8_0] = true, [GGML_TYPE_Q8_1] = true,
    [GGML_TYPE_Q2_K] = true, [GGML_TYPE_Q3_K] = true, [GGML_TYPE_Q4_K] = true, [GGML_TYPE_Q5_K] = true,
    [GGML_TYPE_Q6_K] = true, [GGML_TYPE_Q8_K] = true, [GGML_TYPE_I8] = false,  [GGML_TYPE_I16] = false,
    [GGML_TYPE_I32] = false,
};
static_assert(GGML_TYPE_COUNT == 19, "GGML_IS_QUANTIZED is outdated");

static const char *GGML_OP_NAME[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "REPEAT",
    "REPEAT_BACK",
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "TANH",
    "ELU",
    "RELU",
    "GELU",
    "GELU_QUICK",
    "SILU",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",

    "MUL_MAT",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "ALIBI",
    "CLAMP",
    "CONV_1D",
    "CONV_2D",
    "POOL_1D",
    "POOL_2D",

    "FLASH_ATTN",
    "FLASH_FF",
    "FLASH_ATTN_BACK",
    "WIN_PART",
    "WIN_UNPART",

    "MAP_UNARY",
    "MAP_BINARY",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
};

static_assert(GGML_OP_COUNT == 68, "GGML_OP_COUNT != 68");

static const char *GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "Σx",
    "Σx_k",
    "Σx/n",
    "argmax(x)",
    "repeat(x)",
    "repeat_back(x)",
    "abs(x)",
    "sgn(x)",
    "-x",
    "step(x)",
    "tanh(x)",
    "elu(x)",
    "relu(x)",
    "gelu(x)",
    "gelu_quick(x)",
    "silu(x)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",

    "X*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "alibi(x)",
    "clamp(x)",
    "conv_1d(x)",
    "conv_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",

    "flash_attn(x)",
    "flash_ff(x)",
    "flash_attn_back(x)",
    "win_part(x)",
    "win_unpart(x)",

    "f(x)",
    "f(x,y)",

    "custom(x)",
    "custom(x,y)",
    "custom(x,y,z)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
};

static_assert(GGML_OP_COUNT == 68, "GGML_OP_COUNT != 68");

static_assert(GGML_OP_POOL_COUNT == 2, "GGML_OP_POOL_COUNT != 2");

static_assert(sizeof(struct ggml_object) % GGML_MEM_ALIGN == 0,
              "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(sizeof(struct ggml_tensor) % GGML_MEM_ALIGN == 0,
              "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");

// WARN:
// Mis-confguration can lead to problem that's hard to reason about:
// * At best  it crash or talks nosense.
// * At worst it talks slightly difference but hard to perceive.
//
// An op has to enable INIT or FINALIZE when any of it's branch needs that pass.
// Take care about compile options (e.g., GGML_USE_xxx).
static bool GGML_OP_HAS_INIT[GGML_OP_COUNT] = {0};
static bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = {0};

static void ggml_setup_op_has_task_pass(void) {
  {  // INIT
    bool *p = GGML_OP_HAS_INIT;

    p[GGML_OP_ACC] = true;
    p[GGML_OP_MUL_MAT] = true;
    p[GGML_OP_OUT_PROD] = true;
    p[GGML_OP_SET] = true;
    p[GGML_OP_GET_ROWS_BACK] = true;
    p[GGML_OP_DIAG_MASK_INF] = true;
    p[GGML_OP_DIAG_MASK_ZERO] = true;
    p[GGML_OP_CONV_1D] = true;
    p[GGML_OP_CONV_2D] = true;
    p[GGML_OP_FLASH_ATTN_BACK] = true;
    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
  }

  {  // FINALIZE
    bool *p = GGML_OP_HAS_FINALIZE;

    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
  }
}

//
// ggml context
//

struct ggml_context {
  size_t mem_size;
  void *mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;
  bool no_alloc_save;  // this is used to save the no_alloc state when using scratch buffers

  int n_objects;

  struct ggml_object *objects_begin;
  struct ggml_object *objects_end;

  struct ggml_scratch scratch;
  struct ggml_scratch scratch_save;
};

struct ggml_context_container {
  bool used;

  struct ggml_context context;
};

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
  uint32_t cpus[GGML_NUMA_MAX_CPUS];  // hardware threads on this node
  uint32_t n_cpus;
};

struct ggml_numa_nodes {
  struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
  uint32_t n_nodes;
  uint32_t total_cpus;  // hardware threads on system
};

//
// ggml state
//

struct ggml_state {
  struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
  struct ggml_numa_nodes numa;
};

// global state
static struct ggml_state g_state;
static atomic_int g_state_barrier = 0;

// barrier via spin lock
inline static void ggml_critical_section_start(void) {
  int processing = atomic_fetch_add(&g_state_barrier, 1);

  while (processing > 0) {
    // wait for other threads to finish
    atomic_fetch_sub(&g_state_barrier, 1);
    sched_yield();  // TODO: reconsider this
    processing = atomic_fetch_add(&g_state_barrier, 1);
  }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void ggml_critical_section_end(void) { atomic_fetch_sub(&g_state_barrier, 1); }

void ggml_numa_init(void) {
  if (g_state.numa.n_nodes > 0) {
    fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

    return;
  }

  // TODO
}

bool ggml_is_numa(void) { return g_state.numa.n_nodes > 1; }

////////////////////////////////////////////////////////////////////////////////

void ggml_print_object(const struct ggml_object *obj) {
  printf(" - ggml_object: offset = %zu, size = %zu, next = %p\n", obj->offs, obj->size, (const void *)obj->next);
}

void ggml_print_objects(const struct ggml_context *ctx) {
  struct ggml_object *obj = ctx->objects_begin;

  printf("%s: objects in context %p:\n", __func__, (const void *)ctx);

  while (obj != NULL) {
    ggml_print_object(obj);
    obj = obj->next;
  }

  printf("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(const struct ggml_tensor *tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

int64_t ggml_nrows(const struct ggml_tensor *tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor *tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  // this should handle cases where the tensor is not contiguous in memory
  // probaby just:
  //
  //     return tensor->ne[3]*tensor->nb[3]
  //
  // is enough, but just in case, adding the second part

  return MAX(tensor->ne[3] * tensor->nb[3],
             (ggml_nelements(tensor) * GGML_TYPE_SIZE[tensor->type]) / GGML_BLCK_SIZE[tensor->type]);
}

size_t ggml_nbytes_split(const struct ggml_tensor *tensor, int nrows_split) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (nrows_split * tensor->ne[0] * GGML_TYPE_SIZE[tensor->type]) / GGML_BLCK_SIZE[tensor->type];
}

int ggml_blck_size(enum ggml_type type) { return GGML_BLCK_SIZE[type]; }

size_t ggml_type_size(enum ggml_type type) { return GGML_TYPE_SIZE[type]; }

float ggml_type_sizef(enum ggml_type type) { return ((float)(GGML_TYPE_SIZE[type])) / GGML_BLCK_SIZE[type]; }

const char *ggml_type_name(enum ggml_type type) { return GGML_TYPE_NAME[type]; }

const char *ggml_op_name(enum ggml_op op) { return GGML_OP_NAME[op]; }

size_t ggml_element_size(const struct ggml_tensor *tensor) { return GGML_TYPE_SIZE[tensor->type]; }

bool ggml_is_quantized(enum ggml_type type) { return GGML_IS_QUANTIZED[type]; }

enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype) {
  enum ggml_type wtype = GGML_TYPE_COUNT;

  switch (ftype) {
    case GGML_FTYPE_ALL_F32:
      wtype = GGML_TYPE_F32;
      break;
    case GGML_FTYPE_MOSTLY_F16:
      wtype = GGML_TYPE_F16;
      break;
    case GGML_FTYPE_MOSTLY_Q4_0:
      wtype = GGML_TYPE_Q4_0;
      break;
    case GGML_FTYPE_MOSTLY_Q4_1:
      wtype = GGML_TYPE_Q4_1;
      break;
    case GGML_FTYPE_MOSTLY_Q5_0:
      wtype = GGML_TYPE_Q5_0;
      break;
    case GGML_FTYPE_MOSTLY_Q5_1:
      wtype = GGML_TYPE_Q5_1;
      break;
    case GGML_FTYPE_MOSTLY_Q8_0:
      wtype = GGML_TYPE_Q8_0;
      break;
    case GGML_FTYPE_MOSTLY_Q2_K:
      wtype = GGML_TYPE_Q2_K;
      break;
    case GGML_FTYPE_MOSTLY_Q3_K:
      wtype = GGML_TYPE_Q3_K;
      break;
    case GGML_FTYPE_MOSTLY_Q4_K:
      wtype = GGML_TYPE_Q4_K;
      break;
    case GGML_FTYPE_MOSTLY_Q5_K:
      wtype = GGML_TYPE_Q5_K;
      break;
    case GGML_FTYPE_MOSTLY_Q6_K:
      wtype = GGML_TYPE_Q6_K;
      break;
    case GGML_FTYPE_UNKNOWN:
    case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
      wtype = GGML_TYPE_COUNT;
      break;
  }

  return wtype;
}

size_t ggml_tensor_overhead(void) { return GGML_OBJECT_SIZE + GGML_TENSOR_SIZE + 16; }

bool ggml_is_transposed(const struct ggml_tensor *tensor) { return tensor->nb[0] > tensor->nb[1]; }

bool ggml_is_contiguous(const struct ggml_tensor *tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
         tensor->nb[1] == (tensor->nb[0] * tensor->ne[0]) / GGML_BLCK_SIZE[tensor->type] &&
         tensor->nb[2] == tensor->nb[1] * tensor->ne[1] && tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

bool ggml_is_permuted(const struct ggml_tensor *tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

static inline bool ggml_are_same_shape(const struct ggml_tensor *t0, const struct ggml_tensor *t1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && (t0->ne[1] == t1->ne[1]) && (t0->ne[2] == t1->ne[2]) && (t0->ne[3] == t1->ne[3]);
}

static inline int ggml_up32(int n) { return (n + 31) & ~31; }

static inline int ggml_up(int n, int m) { return (n + m - 1) & ~(m - 1); }

////////////////////////////////////////////////////////////////////////////////

struct ggml_context *ggml_init(struct ggml_init_params params) {
  // make this function thread safe
  ggml_critical_section_start();

  static bool is_first_call = true;

  if (is_first_call) {
    // initialize time system (required on Windows)
    ggml_time_init();

    // initialize GELU, Quick GELU, SILU and EXP F32 tables
    {
      const uint64_t t_start = ggml_time_us();
      UNUSED(t_start);

      ggml_fp16_t ii;
      for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
        table_gelu_f16[i] = ggml_gelu_f32(f);
        table_gelu_quick_f16[i] = ggml_gelu_quick_f32(f);
        table_silu_f16[i] = ggml_silu_f32(f);
        table_exp_f16[i] = expf(f);
      }

      const uint64_t t_end = ggml_time_us();
      UNUSED(t_end);

      GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__,
                       (t_end - t_start) / 1000.0f);
    }

    // initialize g_state
    {
      const uint64_t t_start = ggml_time_us();
      UNUSED(t_start);

      g_state = (struct ggml_state){
          /*.contexts =*/{{0}},
          /*.numa =*/
          {
              .n_nodes = 0,
              .total_cpus = 0,
          },
      };

      for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
        g_state.contexts[i].used = false;
      }

      const uint64_t t_end = ggml_time_us();
      UNUSED(t_end);

      GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0f);
    }

    ggml_setup_op_has_task_pass();

    is_first_call = false;
  }

  // find non-used context in g_state
  struct ggml_context *ctx = NULL;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (!g_state.contexts[i].used) {
      g_state.contexts[i].used = true;
      ctx = &g_state.contexts[i].context;

      GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
      break;
    }
  }

  if (ctx == NULL) {
    GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

    ggml_critical_section_end();

    return NULL;
  }

  const size_t mem_size = (params.mem_size + GGML_MEM_ALIGN - 1) & ~(GGML_MEM_ALIGN - 1);

  *ctx = (struct ggml_context){
      /*.mem_size           =*/mem_size,
      /*.mem_buffer         =*/params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
      /*.mem_buffer_owned   =*/params.mem_buffer ? false : true,
      /*.no_alloc           =*/params.no_alloc,
      /*.no_alloc_save      =*/params.no_alloc,
      /*.n_objects          =*/0,
      /*.objects_begin      =*/NULL,
      /*.objects_end        =*/NULL,
      /*.scratch            =*/
      {
          0,
          0,
          NULL,
      },
      /*.scratch_save       =*/
      {
          0,
          0,
          NULL,
      },
  };

  GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

  ggml_critical_section_end();

  return ctx;
}

void ggml_free(struct ggml_context *ctx) {
  // make this function thread safe
  ggml_critical_section_start();

  bool found = false;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (&g_state.contexts[i].context == ctx) {
      g_state.contexts[i].used = false;

      GGML_PRINT_DEBUG("%s: context %d with %d objects has been freed. memory used = %zu\n", __func__, i,
                       ctx->n_objects, ctx->objects_end->offs + ctx->objects_end->size);

      if (ctx->mem_buffer_owned) {
        free(ctx->mem_buffer);
      }

      found = true;
      break;
    }
  }

  if (!found) {
    GGML_PRINT_DEBUG("%s: context not found\n", __func__);
  }

  ggml_critical_section_end();
}

size_t ggml_used_mem(const struct ggml_context *ctx) {
  return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

size_t ggml_set_scratch(struct ggml_context *ctx, struct ggml_scratch scratch) {
  const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

  ctx->scratch = scratch;

  return result;
}

void ggml_set_no_alloc(struct ggml_context *ctx, bool no_alloc) { ctx->no_alloc = no_alloc; }

void *ggml_get_mem_buffer(const struct ggml_context *ctx) { return ctx->mem_buffer; }

size_t ggml_get_mem_size(const struct ggml_context *ctx) { return ctx->mem_size; }

size_t ggml_get_max_tensor_size(const struct ggml_context *ctx) {
  size_t max_size = 0;

  struct ggml_object *obj = ctx->objects_begin;

  while (obj != NULL) {
    struct ggml_tensor *tensor = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj->offs);

    const size_t size = ggml_nbytes(tensor);

    if (max_size < size) {
      max_size = size;
    }

    obj = obj->next;
  }

  return max_size;
}

// IMPORTANT:
// when creating "opt" tensors, always save and load the scratch buffer
// this is an error prone process, but it is necessary to support inplace
// operators when using scratch buffers
// TODO: implement a better way
void ggml_scratch_save(struct ggml_context *ctx) {
  // this is needed to allow opt tensors to store their data
  // TODO: again, need to find a better way
  ctx->no_alloc_save = ctx->no_alloc;
  ctx->no_alloc = false;

  ctx->scratch_save = ctx->scratch;
  ctx->scratch.data = NULL;
}

void ggml_scratch_load(struct ggml_context *ctx) {
  ctx->no_alloc = ctx->no_alloc_save;

  ctx->scratch = ctx->scratch_save;
}

////////////////////////////////////////////////////////////////////////////////

struct ggml_tensor *ggml_new_tensor_impl(struct ggml_context *ctx, enum ggml_type type, int n_dims, const int64_t *ne,
                                         void *data) {
  // always insert objects at the end of the context's memory pool
  struct ggml_object *obj_cur = ctx->objects_end;

  const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
  const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  const size_t cur_end = cur_offs + cur_size;

  size_t size_needed = 0;

  if (data == NULL && !ctx->no_alloc) {
    size_needed += GGML_TYPE_SIZE[type] * (ne[0] / GGML_BLCK_SIZE[type]);
    for (int i = 1; i < n_dims; i++) {
      size_needed *= ne[i];
    }
    // align to GGML_MEM_ALIGN
    size_needed = ((size_needed + GGML_MEM_ALIGN - 1) / GGML_MEM_ALIGN) * GGML_MEM_ALIGN;
  }

  char *const mem_buffer = ctx->mem_buffer;
  struct ggml_object *const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

  if (ctx->scratch.data == NULL || data != NULL) {
    size_needed += GGML_TENSOR_SIZE;

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
      printf("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__,
             cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);

      return NULL;
    }

    *obj_new = (struct ggml_object){
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
    };
  } else {
    if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
      printf("%s: not enough space in the scratch memory pool (needed %zu, available %zu)\n", __func__,
             ctx->scratch.offs + size_needed, ctx->scratch.size);

      return NULL;
    }

    if (cur_end + GGML_TENSOR_SIZE + GGML_OBJECT_SIZE > ctx->mem_size) {
      printf("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__,
             cur_end + GGML_TENSOR_SIZE + GGML_OBJECT_SIZE, ctx->mem_size);

      return NULL;
    }

    data = (char *const)ctx->scratch.data + ctx->scratch.offs;

    *obj_new = (struct ggml_object){
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = GGML_TENSOR_SIZE,
        .next = NULL,
    };

    // printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

    ctx->scratch.offs += size_needed;
  }

  if (obj_cur != NULL) {
    obj_cur->next = obj_new;
  } else {
    // this is the first object in this context
    ctx->objects_begin = obj_new;
  }

  ctx->objects_end = obj_new;

  // printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

  struct ggml_tensor *const result = (struct ggml_tensor *)(mem_buffer + obj_new->offs);

  *result = (struct ggml_tensor){
      /*.type         =*/type,
      /*.backend      =*/GGML_BACKEND_CPU,
      /*.n_dims       =*/n_dims,
      /*.ne           =*/{1, 1, 1, 1},
      /*.nb           =*/{0, 0, 0, 0},
      /*.op           =*/GGML_OP_NONE,
      /*.is_param     =*/false,
      /*.grad         =*/NULL,
      /*.src          =*/{NULL},
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
      /*.data         =*/(data == NULL && !ctx->no_alloc) ? (void *)(result + 1) : data,
      /*.name         =*/{0},
      /*.extra        =*/NULL,
      /*.padding      =*/{0},
  };

  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }

  result->nb[0] = GGML_TYPE_SIZE[type];
  result->nb[1] = result->nb[0] * (result->ne[0] / GGML_BLCK_SIZE[type]);
  for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }

  ctx->n_objects++;

  return result;
}

struct ggml_tensor *ggml_new_tensor(struct ggml_context *ctx, enum ggml_type type, int n_dims, const int64_t *ne) {
  return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct ggml_tensor *ggml_new_tensor_1d(struct ggml_context *ctx, enum ggml_type type, int64_t ne0) {
  return ggml_new_tensor(ctx, type, 1, &ne0);
}

struct ggml_tensor *ggml_new_tensor_2d(struct ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1) {
  const int64_t ne[2] = {ne0, ne1};
  return ggml_new_tensor(ctx, type, 2, ne);
}

struct ggml_tensor *ggml_new_tensor_3d(struct ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1,
                                       int64_t ne2) {
  const int64_t ne[3] = {ne0, ne1, ne2};
  return ggml_new_tensor(ctx, type, 3, ne);
}

struct ggml_tensor *ggml_new_tensor_4d(struct ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1,
                                       int64_t ne2, int64_t ne3) {
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  return ggml_new_tensor(ctx, type, 4, ne);
}

struct ggml_tensor *ggml_new_i32(struct ggml_context *ctx, int32_t value) {
  ggml_scratch_save(ctx);

  struct ggml_tensor *result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

  ggml_scratch_load(ctx);

  ggml_set_i32(result, value);

  return result;
}

struct ggml_tensor *ggml_new_f32(struct ggml_context *ctx, float value) {
  ggml_scratch_save(ctx);

  struct ggml_tensor *result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

  ggml_scratch_load(ctx);

  ggml_set_f32(result, value);

  return result;
}

struct ggml_tensor *ggml_dup_tensor(struct ggml_context *ctx, const struct ggml_tensor *src) {
  return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}

struct ggml_tensor *ggml_set_zero(struct ggml_tensor *tensor) {
  memset(tensor->data, 0, ggml_nbytes(tensor));
  return tensor;
}

struct ggml_tensor *ggml_set_i32(struct ggml_tensor *tensor, int32_t value) {
  const int n = ggml_nrows(tensor);
  const int nc = tensor->ne[0];
  const size_t n1 = tensor->nb[1];

  char *const data = tensor->data;

  switch (tensor->type) {
    case GGML_TYPE_I8: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i8(nc, (int8_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I16: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i16(nc, (int16_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I32: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i32(nc, (int32_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_F16: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_F32: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f32(nc, (float *)(data + i * n1), value);
      }
    } break;
    default: {
    } break;
  }

  return tensor;
}

struct ggml_tensor *ggml_set_f32(struct ggml_tensor *tensor, float value) {
  const int n = ggml_nrows(tensor);
  const int nc = tensor->ne[0];
  const size_t n1 = tensor->nb[1];

  char *const data = tensor->data;

  switch (tensor->type) {
    case GGML_TYPE_I8: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i8(nc, (int8_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I16: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i16(nc, (int16_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I32: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i32(nc, (int32_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_F16: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_F32: {
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f32(nc, (float *)(data + i * n1), value);
      }
    } break;
    default: {
    } break;
  }

  return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor *tensor, int i) {
  switch (tensor->type) {
    case GGML_TYPE_I8: {
      return ((int8_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_I16: {
      return ((int16_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_I32: {
      return ((int32_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_F16: {
      return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
    } break;
    case GGML_TYPE_F32: {
      return ((float *)(tensor->data))[i];
    } break;
    default: {
    } break;
  }

  return 0.0f;
}

void ggml_set_i32_1d(const struct ggml_tensor *tensor, int i, int32_t value) {
  switch (tensor->type) {
    case GGML_TYPE_I8: {
      ((int8_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_I16: {
      ((int16_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_I32: {
      ((int32_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_F16: {
      ((ggml_fp16_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_F32: {
      ((float *)(tensor->data))[i] = value;
    } break;
    default: {
    } break;
  }
}

float ggml_get_f32_1d(const struct ggml_tensor *tensor, int i) {
  switch (tensor->type) {
    case GGML_TYPE_I8: {
      return ((int8_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_I16: {
      return ((int16_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_I32: {
      return ((int32_t *)(tensor->data))[i];
    } break;
    case GGML_TYPE_F16: {
      return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
    } break;
    case GGML_TYPE_F32: {
      return ((float *)(tensor->data))[i];
    } break;
    default: {
    } break;
  }

  return 0.0f;
}

void ggml_set_f32_1d(const struct ggml_tensor *tensor, int i, float value) {
  switch (tensor->type) {
    case GGML_TYPE_I8: {
      ((int8_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_I16: {
      ((int16_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_I32: {
      ((int32_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_F16: {
      ((ggml_fp16_t *)(tensor->data))[i] = value;
    } break;
    case GGML_TYPE_F32: {
      ((float *)(tensor->data))[i] = value;
    } break;
    default: {
    } break;
  }
}

void *ggml_get_data(const struct ggml_tensor *tensor) { return tensor->data; }

float *ggml_get_data_f32(const struct ggml_tensor *tensor) { return (float *)(tensor->data); }

const char *ggml_get_name(const struct ggml_tensor *tensor) { return tensor->name; }

struct ggml_tensor *ggml_set_name(struct ggml_tensor *tensor, const char *name) {
  strncpy(tensor->name, name, sizeof(tensor->name));
  tensor->name[sizeof(tensor->name) - 1] = '\0';
  return tensor;
}

struct ggml_tensor *ggml_format_name(struct ggml_tensor *tensor, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
  va_end(args);
  return tensor;
}

struct ggml_tensor *ggml_view_tensor(struct ggml_context *ctx, const struct ggml_tensor *src) {
  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);
  ggml_format_name(result, "%s (view)", src->name);

  result->nb[0] = src->nb[0];
  result->nb[1] = src->nb[1];
  result->nb[2] = src->nb[2];
  result->nb[3] = src->nb[3];

  return result;
}

struct ggml_tensor *ggml_get_tensor(struct ggml_context *ctx, const char *name) {
  struct ggml_object *obj = ctx->objects_begin;

  char *const mem_buffer = ctx->mem_buffer;

  while (obj != NULL) {
    struct ggml_tensor *cur = (struct ggml_tensor *)(mem_buffer + obj->offs);
    if (strcmp(cur->name, name) == 0) {
      return cur;
    }

    obj = obj->next;
  }

  return NULL;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

struct ggml_tensor *ggml_dup_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_DUP;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_dup(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_dup_impl(ctx, a, false); }

struct ggml_tensor *ggml_dup_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_dup_impl(ctx, a, true);
}

// ggml_add

struct ggml_tensor *ggml_add_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                  bool inplace) {
  // TODO: support less-strict constraint

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    // TODO: support backward pass for broadcasting

    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_ADD;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_add(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_add_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_add_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_add_impl(ctx, a, b, true);
}

// ggml_add1

struct ggml_tensor *ggml_add1_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                   bool inplace) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_ADD1;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_add1(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_add1_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_add1_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

struct ggml_tensor *ggml_acc_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                                  size_t nb2, size_t nb3, size_t offset, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);

  ((int32_t *)c->data)[0] = nb1;
  ((int32_t *)c->data)[1] = nb2;
  ((int32_t *)c->data)[2] = nb3;
  ((int32_t *)c->data)[3] = offset;
  ((int32_t *)c->data)[4] = inplace ? 1 : 0;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_ACC;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

struct ggml_tensor *ggml_acc(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                             size_t nb2, size_t nb3, size_t offset) {
  return ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct ggml_tensor *ggml_acc_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                                     size_t nb2, size_t nb3, size_t offset) {
  return ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

struct ggml_tensor *ggml_sub_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                  bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SUB;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_sub(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_sub_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_sub_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

struct ggml_tensor *ggml_mul_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                  bool inplace) {
  // TODO: support less-strict constraint

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    // TODO: support backward pass for broadcasting

    is_node = true;
  }

  if (inplace) {
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_MUL;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_mul(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_mul_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_mul_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

struct ggml_tensor *ggml_div_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                  bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  if (inplace) {
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_DIV;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_div(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_div_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_div_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

struct ggml_tensor *ggml_sqr_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SQR;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_sqr(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_sqr_impl(ctx, a, false); }

struct ggml_tensor *ggml_sqr_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

struct ggml_tensor *ggml_sqrt_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SQRT;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_sqrt(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_sqrt_impl(ctx, a, false); }

struct ggml_tensor *ggml_sqrt_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

struct ggml_tensor *ggml_log_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_LOG;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_log(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_log_impl(ctx, a, false); }

struct ggml_tensor *ggml_log_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_log_impl(ctx, a, true);
}

// ggml_sum

struct ggml_tensor *ggml_sum(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_new_tensor_1d(ctx, a->type, 1);

  result->op = GGML_OP_SUM;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_sum_rows

struct ggml_tensor *ggml_sum_rows(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  int64_t ne[4] = {1, 1, 1, 1};
  for (int i = 1; i < a->n_dims; ++i) {
    ne[i] = a->ne[i];
  }

  struct ggml_tensor *result = ggml_new_tensor(ctx, a->type, a->n_dims, ne);

  result->op = GGML_OP_SUM_ROWS;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_mean

struct ggml_tensor *ggml_mean(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement
    is_node = true;
  }

  int64_t ne[GGML_MAX_DIMS] = {1, a->ne[1], a->ne[2], a->ne[3]};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, a->n_dims, ne);

  result->op = GGML_OP_MEAN;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_argmax

struct ggml_tensor *ggml_argmax(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  int64_t ne[GGML_MAX_DIMS] = {a->ne[1], 1, 1, 1};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_I32, a->n_dims, ne);

  result->op = GGML_OP_ARGMAX;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_repeat

struct ggml_tensor *ggml_repeat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  if (ggml_are_same_shape(a, b) && !is_node) {
    return a;
  }

  struct ggml_tensor *result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

  result->op = GGML_OP_REPEAT;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_repeat_back

struct ggml_tensor *ggml_repeat_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  if (ggml_are_same_shape(a, b) && !is_node) {
    return a;
  }

  struct ggml_tensor *result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

  result->op = GGML_OP_REPEAT_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_abs

struct ggml_tensor *ggml_abs_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_ABS;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_abs(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_abs_impl(ctx, a, false); }

struct ggml_tensor *ggml_abs_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_abs_impl(ctx, a, true);
}

// ggml_sgn

struct ggml_tensor *ggml_sgn_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SGN;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_sgn(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_sgn_impl(ctx, a, false); }

struct ggml_tensor *ggml_sgn_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_sgn_impl(ctx, a, true);
}

// ggml_neg

struct ggml_tensor *ggml_neg_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_NEG;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_neg(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_neg_impl(ctx, a, false); }

struct ggml_tensor *ggml_neg_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_neg_impl(ctx, a, true);
}

// ggml_step

struct ggml_tensor *ggml_step_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_STEP;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_step(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_step_impl(ctx, a, false); }

struct ggml_tensor *ggml_step_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_step_impl(ctx, a, true);
}

// ggml_tanh

struct ggml_tensor *ggml_tanh_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_TANH;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_tanh(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_tanh_impl(ctx, a, false); }

struct ggml_tensor *ggml_tanh_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_tanh_impl(ctx, a, true);
}

// ggml_elu

struct ggml_tensor *ggml_elu_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_ELU;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_elu(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_elu_impl(ctx, a, false); }

struct ggml_tensor *ggml_elu_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_elu_impl(ctx, a, true);
}

// ggml_relu

struct ggml_tensor *ggml_relu_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_RELU;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_relu(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_relu_impl(ctx, a, false); }

struct ggml_tensor *ggml_relu_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_relu_impl(ctx, a, true);
}

// ggml_gelu

struct ggml_tensor *ggml_gelu_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_GELU;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_gelu(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_gelu_impl(ctx, a, false); }

struct ggml_tensor *ggml_gelu_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_gelu_impl(ctx, a, true);
}

// ggml_gelu_quick

struct ggml_tensor *ggml_gelu_quick_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_GELU_QUICK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_gelu_quick(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_gelu_quick_impl(ctx, a, false);
}

struct ggml_tensor *ggml_gelu_quick_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_gelu_quick_impl(ctx, a, true);
}

// ggml_silu

struct ggml_tensor *ggml_silu_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SILU;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_silu(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_silu_impl(ctx, a, false); }

struct ggml_tensor *ggml_silu_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_silu_impl(ctx, a, true);
}

// ggml_silu_back

struct ggml_tensor *ggml_silu_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    // TODO: implement backward
    is_node = true;
  }

  struct ggml_tensor *result = ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SILU_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_norm

struct ggml_tensor *ggml_norm_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_NORM;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;  // TODO: maybe store epsilon here?

  return result;
}

struct ggml_tensor *ggml_norm(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_norm_impl(ctx, a, false); }

struct ggml_tensor *ggml_norm_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_norm_impl(ctx, a, true);
}

struct ggml_tensor *ggml_rms_norm_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_RMS_NORM;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;  // TODO: maybe store epsilon here?

  return result;
}

struct ggml_tensor *ggml_rms_norm(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_rms_norm_impl(ctx, a, false);
}

struct ggml_tensor *ggml_rms_norm_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_rms_norm_impl(ctx, a, true);
}

struct ggml_tensor *ggml_rms_norm_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad) {
    // TODO: implement backward
    is_node = true;
  }

  struct ggml_tensor *result = ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_RMS_NORM_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_mul_mat

struct ggml_tensor *ggml_mul_mat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, MAX(a->n_dims, b->n_dims), ne);

  result->op = GGML_OP_MUL_MAT;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_out_prod

struct ggml_tensor *ggml_out_prod(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[0], b->ne[0], a->ne[2], b->ne[3]};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, MIN(a->n_dims, b->n_dims), ne);

  result->op = GGML_OP_OUT_PROD;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_scale

struct ggml_tensor *ggml_scale_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                    bool inplace) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SCALE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_scale(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_scale_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_scale_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_scale_impl(ctx, a, b, true);
}

// ggml_set

struct ggml_tensor *ggml_set_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                                  size_t nb2, size_t nb3, size_t offset, bool inplace) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // make a view of the destination
  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);

  ((int32_t *)c->data)[0] = nb1;
  ((int32_t *)c->data)[1] = nb2;
  ((int32_t *)c->data)[2] = nb3;
  ((int32_t *)c->data)[3] = offset;
  ((int32_t *)c->data)[4] = inplace ? 1 : 0;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_SET;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

struct ggml_tensor *ggml_set(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                             size_t nb2, size_t nb3, size_t offset) {
  return ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct ggml_tensor *ggml_set_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                                     size_t nb2, size_t nb3, size_t offset) {
  return ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

struct ggml_tensor *ggml_set_1d(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t offset) {
  return ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

struct ggml_tensor *ggml_set_1d_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                        size_t offset) {
  return ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

struct ggml_tensor *ggml_set_2d(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, size_t nb1,
                                size_t offset) {
  return ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

struct ggml_tensor *ggml_set_2d_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                        size_t nb1, size_t offset) {
  return ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

// ggml_cpy

struct ggml_tensor *ggml_cpy_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                  bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  // make a view of the destination
  struct ggml_tensor *result = ggml_view_tensor(ctx, b);
  if (strlen(b->name) > 0) {
    ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
  } else {
    ggml_format_name(result, "%s (copy)", a->name);
  }

  result->op = GGML_OP_CPY;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_cpy(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_cpy_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_cpy_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_cpy_impl(ctx, a, b, true);
}

// ggml_cont

struct ggml_tensor *ggml_cont_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
  ggml_format_name(result, "%s (cont)", a->name);

  result->op = GGML_OP_CONT;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_cont(struct ggml_context *ctx, struct ggml_tensor *a) { return ggml_cont_impl(ctx, a, false); }

struct ggml_tensor *ggml_cont_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_cont_impl(ctx, a, true);
}

// ggml_reshape

struct ggml_tensor *ggml_reshape(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_reshape_1d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[1] = {ne0};
  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a->data);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_reshape_2d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[2] = {ne0, ne1};
  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a->data);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_reshape_3d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1,
                                    int64_t ne2) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[3] = {ne0, ne1, ne2};
  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a->data);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_reshape_4d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1,
                                    int64_t ne2, int64_t ne3) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a->data);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_view_1d

struct ggml_tensor *ggml_view_1d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 1, &ne0, (char *)a->data + offset);
  ggml_format_name(result, "%s (view)", a->name);

  ggml_scratch_save(ctx);

  struct ggml_tensor *offs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
  ggml_set_name(offs, "offset");
  memcpy(offs->data, &offset, 2 * sizeof(int32_t));

  ggml_scratch_load(ctx);

  result->op = GGML_OP_VIEW;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = offs;

  return result;
}

// ggml_view_2d

struct ggml_tensor *ggml_view_2d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1, size_t nb1,
                                 size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, 1, 1};

  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 2, ne, (char *)a->data + offset);
  ggml_format_name(result, "%s (view)", a->name);

  ggml_scratch_save(ctx);

  struct ggml_tensor *offs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
  ggml_set_name(offs, "offset");
  memcpy(offs->data, &offset, 2 * sizeof(int32_t));

  ggml_scratch_load(ctx);

  result->nb[1] = nb1;
  result->nb[2] = result->nb[1] * ne1;
  result->nb[3] = result->nb[2];

  result->op = GGML_OP_VIEW;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = offs;

  return result;
}

// ggml_view_3d

struct ggml_tensor *ggml_view_3d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2,
                                 size_t nb1, size_t nb2, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, ne2, 1};

  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 3, ne, (char *)a->data + offset);
  ggml_format_name(result, "%s (view)", a->name);

  ggml_scratch_save(ctx);

  struct ggml_tensor *offs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
  ggml_set_name(offs, "offset");
  memcpy(offs->data, &offset, 2 * sizeof(int32_t));

  ggml_scratch_load(ctx);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = result->nb[2] * ne2;

  result->op = GGML_OP_VIEW;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = offs;

  return result;
}

// ggml_view_4d

struct ggml_tensor *ggml_view_4d(struct ggml_context *ctx, struct ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2,
                                 int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, ne2, ne3};

  struct ggml_tensor *result = ggml_new_tensor_impl(ctx, a->type, 4, ne, (char *)a->data + offset);
  ggml_format_name(result, "%s (view)", a->name);

  ggml_scratch_save(ctx);

  struct ggml_tensor *offs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
  ggml_set_name(offs, "offset");
  memcpy(offs->data, &offset, 2 * sizeof(int32_t));

  ggml_scratch_load(ctx);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = nb3;

  result->op = GGML_OP_VIEW;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = offs;

  return result;
}

// ggml_permute

struct ggml_tensor *ggml_permute(struct ggml_context *ctx, struct ggml_tensor *a, int axis0, int axis1, int axis2,
                                 int axis3) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_view_tensor(ctx, a);
  ggml_format_name(result, "%s (permuted)", a->name);

  int ne[GGML_MAX_DIMS];
  int nb[GGML_MAX_DIMS];

  ne[axis0] = a->ne[0];
  ne[axis1] = a->ne[1];
  ne[axis2] = a->ne[2];
  ne[axis3] = a->ne[3];

  nb[axis0] = a->nb[0];
  nb[axis1] = a->nb[1];
  nb[axis2] = a->nb[2];
  nb[axis3] = a->nb[3];

  result->ne[0] = ne[0];
  result->ne[1] = ne[1];
  result->ne[2] = ne[2];
  result->ne[3] = ne[3];

  result->nb[0] = nb[0];
  result->nb[1] = nb[1];
  result->nb[2] = nb[2];
  result->nb[3] = nb[3];

  result->op = GGML_OP_PERMUTE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  if (is_node) {
    ggml_scratch_save(ctx);

    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);

    ((int32_t *)b->data)[0] = axis0;
    ((int32_t *)b->data)[1] = axis1;
    ((int32_t *)b->data)[2] = axis2;
    ((int32_t *)b->data)[3] = axis3;

    ggml_scratch_load(ctx);

    result->src[2] = b;
  }

  return result;
}

// ggml_transpose

struct ggml_tensor *ggml_transpose(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_view_tensor(ctx, a);
  ggml_format_name(result, "%s (transposed)", a->name);

  result->ne[0] = a->ne[1];
  result->ne[1] = a->ne[0];

  result->nb[0] = a->nb[1];
  result->nb[1] = a->nb[0];

  result->op = GGML_OP_TRANSPOSE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_get_rows

struct ggml_tensor *ggml_get_rows(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // TODO: implement non F32 return
  // struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ggml_tensor *result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a->ne[0], b->ne[0]);

  result->op = GGML_OP_GET_ROWS;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_get_rows_back

struct ggml_tensor *ggml_get_rows_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                       struct ggml_tensor *c) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // TODO: implement non F32 return
  // struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ggml_tensor *result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

  result->op = GGML_OP_GET_ROWS_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

// ggml_diag

struct ggml_tensor *ggml_diag(struct ggml_context *ctx, struct ggml_tensor *a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[0], a->ne[0], a->ne[2], a->ne[3]};
  struct ggml_tensor *result = ggml_new_tensor(ctx, a->type, MAX(a->n_dims, 2), ne);

  result->op = GGML_OP_DIAG;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

// ggml_diag_mask_inf

struct ggml_tensor *ggml_diag_mask_inf_impl(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);

  ((int32_t *)b->data)[0] = n_past;
  ((int32_t *)b->data)[1] = inplace ? 1 : 0;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_DIAG_MASK_INF;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_diag_mask_inf(struct ggml_context *ctx, struct ggml_tensor *a, int n_past) {
  return ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

struct ggml_tensor *ggml_diag_mask_inf_inplace(struct ggml_context *ctx, struct ggml_tensor *a, int n_past) {
  return ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

struct ggml_tensor *ggml_diag_mask_zero_impl(struct ggml_context *ctx, struct ggml_tensor *a, int n_past,
                                             bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
  ggml_set_name(b, "n_past, inplace");

  ((int32_t *)b->data)[0] = n_past;
  ((int32_t *)b->data)[1] = inplace ? 1 : 0;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_DIAG_MASK_ZERO;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_diag_mask_zero(struct ggml_context *ctx, struct ggml_tensor *a, int n_past) {
  return ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

struct ggml_tensor *ggml_diag_mask_zero_inplace(struct ggml_context *ctx, struct ggml_tensor *a, int n_past) {
  return ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_soft_max

struct ggml_tensor *ggml_soft_max_impl(struct ggml_context *ctx, struct ggml_tensor *a, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SOFT_MAX;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;

  return result;
}

struct ggml_tensor *ggml_soft_max(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_soft_max_impl(ctx, a, false);
}

struct ggml_tensor *ggml_soft_max_inplace(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_soft_max_impl(ctx, a, true);
}

// ggml_soft_max_back

struct ggml_tensor *ggml_soft_max_back_impl(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                            bool inplace) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;  // TODO : implement backward pass
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_SOFT_MAX_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_soft_max_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_soft_max_back_impl(ctx, a, b, false);
}

struct ggml_tensor *ggml_soft_max_back_inplace(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  return ggml_soft_max_back_impl(ctx, a, b, true);
}

// ggml_rope

struct ggml_tensor *ggml_rope_impl(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, int n_dims, int mode,
                                   int n_ctx, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);

  ((int32_t *)b->data)[0] = n_past;
  ((int32_t *)b->data)[1] = n_dims;
  ((int32_t *)b->data)[2] = mode;
  ((int32_t *)b->data)[3] = n_ctx;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_ROPE;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor *ggml_rope(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, int n_dims, int mode,
                              int n_ctx) {
  return ggml_rope_impl(ctx, a, n_past, n_dims, mode, n_ctx, false);
}

struct ggml_tensor *ggml_rope_inplace(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, int n_dims, int mode,
                                      int n_ctx) {
  return ggml_rope_impl(ctx, a, n_past, n_dims, mode, n_ctx, true);
}

// ggml_rope_back

struct ggml_tensor *ggml_rope_back(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, int n_dims, int mode) {
  bool is_node = false;

  if (a->grad) {
    is_node = false;  // TODO: implement backward
  }

  struct ggml_tensor *result = ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
  ggml_set_name(b, "n_past, n_dims, mode");

  ((int32_t *)b->data)[0] = n_past;
  ((int32_t *)b->data)[1] = n_dims;
  ((int32_t *)b->data)[2] = mode;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_ROPE_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_alibi

struct ggml_tensor *ggml_alibi(struct ggml_context *ctx, struct ggml_tensor *a, int n_past, int n_head,
                               float bias_max) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  // TODO: when implement backward, fix this:
  // struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
  struct ggml_tensor *result = ggml_view_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);

  ((int32_t *)b->data)[0] = n_past;
  ((int32_t *)b->data)[1] = n_head;

  (((float *)b->data)[2]) = bias_max;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_ALIBI;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_clamp

struct ggml_tensor *ggml_clamp(struct ggml_context *ctx, struct ggml_tensor *a, float min, float max) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  // TODO: when implement backward, fix this:
  struct ggml_tensor *result = ggml_view_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

  ((float *)b->data)[0] = min;
  ((float *)b->data)[1] = max;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_CLAMP;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_conv_1d

static int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
  return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

struct ggml_tensor *ggml_conv_1d(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, int s0, int p0,
                                 int d0) {
  bool is_node = false;

  if (a->grad || b->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[4] = {
      ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0),
      a->ne[2],
      1,
      1,
  };
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

  ggml_scratch_save(ctx);
  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
  ((int32_t *)c->data)[0] = s0;
  ((int32_t *)c->data)[1] = p0;
  ((int32_t *)c->data)[2] = d0;
  ggml_scratch_load(ctx);

  result->op = GGML_OP_CONV_1D;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

// ggml_conv_2d

struct ggml_tensor *ggml_conv_2d(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, int s0, int s1,
                                 int p0, int p1, int d0, int d1) {
  bool is_node = false;

  if (a->grad || b->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[4] = {
      ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0),
      ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1),
      a->ne[3],
      b->ne[3],
  };
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  ggml_scratch_save(ctx);
  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
  ((int32_t *)c->data)[0] = s0;
  ((int32_t *)c->data)[1] = s1;
  ((int32_t *)c->data)[2] = p0;
  ((int32_t *)c->data)[3] = p1;
  ((int32_t *)c->data)[4] = d0;
  ((int32_t *)c->data)[5] = d1;
  ggml_scratch_load(ctx);

  result->op = GGML_OP_CONV_2D;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

// ggml_conv_1d_ph

struct ggml_tensor *ggml_conv_1d_ph(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, int s,
                                    int d) {
  return ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_pool_*

static int64_t ggml_calc_pool_output_size(int64_t ins, int ks, int s, int p) { return (ins + 2 * p - ks) / s + 1; }

// ggml_pool_2d

struct ggml_tensor *ggml_pool_1d(struct ggml_context *ctx, struct ggml_tensor *a, enum ggml_op_pool op, int k0, int s0,
                                 int p0) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[3] = {
      ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
      a->ne[1],
  };
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

  ggml_scratch_save(ctx);
  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
  ((int32_t *)c->data)[0] = op;
  ((int32_t *)c->data)[1] = k0;
  ((int32_t *)c->data)[2] = s0;
  ((int32_t *)c->data)[3] = p0;
  ggml_scratch_load(ctx);

  result->op = GGML_OP_POOL_1D;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = c;

  return result;
}

// ggml_pool_2d

struct ggml_tensor *ggml_pool_2d(struct ggml_context *ctx, struct ggml_tensor *a, enum ggml_op_pool op, int k0, int k1,
                                 int s0, int s1, int p0, int p1) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[3] = {
      ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
      ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
      a->ne[2],
  };
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

  ggml_scratch_save(ctx);
  struct ggml_tensor *c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
  ((int32_t *)c->data)[0] = op;
  ((int32_t *)c->data)[1] = k0;
  ((int32_t *)c->data)[2] = k1;
  ((int32_t *)c->data)[3] = s0;
  ((int32_t *)c->data)[4] = s1;
  ((int32_t *)c->data)[5] = p0;
  ((int32_t *)c->data)[6] = p1;
  ggml_scratch_load(ctx);

  result->op = GGML_OP_POOL_2D;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = c;

  return result;
}

// ggml_flash_attn

struct ggml_tensor *ggml_flash_attn(struct ggml_context *ctx, struct ggml_tensor *q, struct ggml_tensor *k,
                                    struct ggml_tensor *v, bool masked) {
  // TODO: check if vT can be multiplied by (k*qT)

  bool is_node = false;

  if (q->grad || k->grad || v->grad) {
    is_node = true;
  }

  // struct ggml_tensor * result = ggml_dup_tensor(ctx, q);
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, q->ne);

  result->op = GGML_OP_FLASH_ATTN;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = q;
  result->src[1] = k;
  result->src[2] = v;
  result->src[3] = ggml_new_i32(ctx, masked ? 1 : 0);

  return result;
}

// ggml_flash_ff

struct ggml_tensor *ggml_flash_ff(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b0,
                                  struct ggml_tensor *b1, struct ggml_tensor *c0, struct ggml_tensor *c1) {
  // TODO: more checks

  bool is_node = false;

  if (a->grad || b0->grad || b1->grad || c0->grad || c1->grad) {
    is_node = true;
  }

  // struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, a->ne);

  result->op = GGML_OP_FLASH_FF;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b0;
  result->src[2] = b1;
  result->src[3] = c0;
  result->src[4] = c1;

  return result;
}

// ggml_flash_attn_back

struct ggml_tensor *ggml_flash_attn_back(struct ggml_context *ctx, struct ggml_tensor *q, struct ggml_tensor *k,
                                         struct ggml_tensor *v, struct ggml_tensor *d, bool masked) {
  // TODO: check if vT can be multiplied by (k*qT)

  // d shape [D,N,ne2,ne3]
  // q shape [D,N,ne2,ne3]
  // k shape [D,M,ne2,ne3]
  // v shape [M,D,ne2,ne3]

  const int64_t D = q->ne[0];
  const int64_t N = q->ne[1];
  const int64_t M = k->ne[1];
  const int64_t ne2 = q->ne[2];
  const int64_t ne3 = q->ne[3];

  bool is_node = false;

  if (q->grad || k->grad || v->grad) {
    // when using this operation (in backwards pass) these grads are set.
    // we don't want to create (big) grad of our result, so is_node is false.
    is_node = false;
  }

  // store gradients of q, k and v as continuous tensors concatenated in result.
  // q shape[D,N,ne2,ne3] ; k shape [D,M,ne2,ne3] ; v shape [M,D,ne2,ne3]
  // gradq->data = result->data
  // gradk->data = result->data + nb0*D*N*ne2*ne3
  // gradv->data = result->data + nb0*D*N*ne2*ne3 + nb0*D*M*ne2*ne3
  // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
  int64_t ne[4] = {D, M + N + M, ne2, ne3};

  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  result->op = GGML_OP_FLASH_ATTN_BACK;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = q;
  result->src[1] = k;
  result->src[2] = v;
  result->src[3] = d;
  result->src[4] = ggml_new_i32(ctx, masked ? 1 : 0);

  return result;
}

// ggml_win_part

struct ggml_tensor *ggml_win_part(struct ggml_context *ctx, struct ggml_tensor *a, int w) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  // padding
  const int px = (w - a->ne[1] % w) % w;
  const int py = (w - a->ne[2] % w) % w;

  const int npx = (px + a->ne[1]) / w;
  const int npy = (py + a->ne[2]) / w;
  const int np = npx * npy;

  const int64_t ne[4] = {
      a->ne[0],
      w,
      w,
      np,
  };

  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);

  ((int32_t *)b->data)[0] = npx;
  ((int32_t *)b->data)[1] = npy;
  ((int32_t *)b->data)[2] = w;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_WIN_PART;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = b;

  return result;
}

// ggml_win_unpart

struct ggml_tensor *ggml_win_unpart(struct ggml_context *ctx, struct ggml_tensor *a, int w0, int h0, int w) {
  bool is_node = false;

  if (a->grad) {
    abort();  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[4] = {
      a->ne[0],
      w0,
      h0,
      1,
  };
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

  ggml_scratch_save(ctx);

  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

  ((int32_t *)b->data)[0] = w;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_WIN_UNPART;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = NULL;
  result->src[2] = b;

  return result;
}

// ggml_map_unary

struct ggml_tensor *ggml_map_unary_impl_f32(struct ggml_context *ctx, struct ggml_tensor *a,
                                            const ggml_unary_op_f32_t fun, bool inplace) {
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_MAP_UNARY;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[2] = addr_tensor;

  return result;
}

struct ggml_tensor *ggml_map_unary_f32(struct ggml_context *ctx, struct ggml_tensor *a, const ggml_unary_op_f32_t fun) {
  return ggml_map_unary_impl_f32(ctx, a, fun, false);
}

struct ggml_tensor *ggml_map_unary_inplace_f32(struct ggml_context *ctx, struct ggml_tensor *a,
                                               const ggml_unary_op_f32_t fun) {
  return ggml_map_unary_impl_f32(ctx, a, fun, true);
}

// ggml_map_binary

struct ggml_tensor *ggml_map_binary_impl_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                             const ggml_binary_op_f32_t fun, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_MAP_BINARY;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = addr_tensor;

  return result;
}

struct ggml_tensor *ggml_map_binary_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                        const ggml_binary_op_f32_t fun) {
  return ggml_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct ggml_tensor *ggml_map_binary_inplace_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                                const ggml_binary_op_f32_t fun) {
  return ggml_map_binary_impl_f32(ctx, a, b, fun, true);
}

// ggml_map_custom1

struct ggml_tensor *ggml_map_custom1_impl_f32(struct ggml_context *ctx, struct ggml_tensor *a,
                                              const ggml_custom1_op_f32_t fun, bool inplace) {
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_MAP_CUSTOM1;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[2] = addr_tensor;

  return result;
}

struct ggml_tensor *ggml_map_custom1_f32(struct ggml_context *ctx, struct ggml_tensor *a,
                                         const ggml_custom1_op_f32_t fun) {
  return ggml_map_custom1_impl_f32(ctx, a, fun, false);
}

struct ggml_tensor *ggml_map_custom1_inplace_f32(struct ggml_context *ctx, struct ggml_tensor *a,
                                                 const ggml_custom1_op_f32_t fun) {
  return ggml_map_custom1_impl_f32(ctx, a, fun, true);
}

// ggml_map_custom2

struct ggml_tensor *ggml_map_custom2_impl_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                              const ggml_custom2_op_f32_t fun, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_MAP_CUSTOM2;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = addr_tensor;

  return result;
}

struct ggml_tensor *ggml_map_custom2_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                         const ggml_custom2_op_f32_t fun) {
  return ggml_map_custom2_impl_f32(ctx, a, b, fun, false);
}

struct ggml_tensor *ggml_map_custom2_inplace_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                                 const ggml_custom2_op_f32_t fun) {
  return ggml_map_custom2_impl_f32(ctx, a, b, fun, true);
}

// ggml_map_custom3

struct ggml_tensor *ggml_map_custom3_impl_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                              struct ggml_tensor *c, const ggml_custom3_op_f32_t fun, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad || b->grad || c->grad)) {
    is_node = true;
  }

  struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

  ggml_scratch_save(ctx);

  struct ggml_tensor *addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;

  ggml_scratch_load(ctx);

  result->op = GGML_OP_MAP_CUSTOM3;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = addr_tensor;
  result->src[3] = c;

  return result;
}

struct ggml_tensor *ggml_map_custom3_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                         struct ggml_tensor *c, const ggml_custom3_op_f32_t fun) {
  return ggml_map_custom3_impl_f32(ctx, a, b, c, fun, false);
}

struct ggml_tensor *ggml_map_custom3_inplace_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                                 struct ggml_tensor *c, const ggml_custom3_op_f32_t fun) {
  return ggml_map_custom3_impl_f32(ctx, a, b, c, fun, true);
}

// ggml_cross_entropy_loss

struct ggml_tensor *ggml_cross_entropy_loss(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  struct ggml_tensor *result = ggml_new_tensor_1d(ctx, a->type, 1);

  result->op = GGML_OP_CROSS_ENTROPY_LOSS;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_cross_entropy_loss_back

struct ggml_tensor *ggml_cross_entropy_loss_back(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b,
                                                 struct ggml_tensor *c) {
  struct ggml_tensor *result = ggml_dup_tensor(ctx, a);

  result->op = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
  result->grad = NULL;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

////////////////////////////////////////////////////////////////////////////////

void ggml_set_param(struct ggml_context *ctx, struct ggml_tensor *tensor) {
  tensor->is_param = true;

  tensor->grad = ggml_dup_tensor(ctx, tensor);
}

// ggml_compute_forward_dup

static void ggml_compute_forward_dup_same_cont(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const size_t nb00 = src0->nb[0];
  const size_t nb0 = dst->nb[0];

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  // parallelize by elements
  const int ne = ggml_nelements(dst);
  const int dr = (ne + nth - 1) / nth;
  const int ie0 = dr * ith;
  const int ie1 = MIN(ie0 + dr, ne);

  if (ie0 < ie1) {
    memcpy(((char *)dst->data + ie0 * nb0), ((char *)src0->data + ie0 * nb00),
           (ie1 - ie0) * GGML_TYPE_SIZE[src0->type]);
  }
}
static void ggml_compute_forward_dup_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == GGML_TYPE_SIZE[src0->type] &&
      nb0 == GGML_TYPE_SIZE[dst->type]) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

  if (ggml_is_contiguous(dst)) {
    if (nb00 == sizeof(ggml_fp16_t)) {
      if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char *dst_ptr = (char *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char *src0_ptr = (char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float *dst_ptr = (float *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              for (int i00 = 0; i00 < ne00; i00++) {
                dst_ptr[id] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (type_traits[dst->type].from_float) {
        ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
        float *src0_f32 = (float *)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / GGML_BLCK_SIZE[dst->type]);
        char *dst_ptr = (char *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

              for (int i00 = 0; i00 < ne00; i00++) {
                src0_f32[i00] = GGML_FP16_TO_FP32(src0_ptr[i00]);
              }

              quantize_row_q(src0_f32, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        abort();  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float *dst_ptr = (float *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ggml_fp16_t *src0_ptr =
                    (ggml_fp16_t *)((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = GGML_FP16_TO_FP32(*src0_ptr);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        ggml_fp16_t *dst_ptr = (ggml_fp16_t *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ggml_fp16_t *src0_ptr =
                    (ggml_fp16_t *)((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        abort();  // TODO: implement
      }
    }
    return;
  }

  // dst counters
  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == GGML_TYPE_F16) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char *src0_ptr = ((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char *dst_ptr = ((char *)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            memcpy(dst_ptr, src0_ptr, sizeof(ggml_fp16_t));

            if (++i10 == ne00) {
              i10 = 0;
              if (++i11 == ne01) {
                i11 = 0;
                if (++i12 == ne02) {
                  i12 = 0;
                  if (++i13 == ne03) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == GGML_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char *src0_ptr = ((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char *dst_ptr = ((char *)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            *(float *)dst_ptr = GGML_FP16_TO_FP32(*(const ggml_fp16_t *)src0_ptr);

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else {
    abort();  // TODO: implement
  }
}

static void ggml_compute_forward_dup_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == GGML_TYPE_SIZE[src0->type] &&
      nb0 == GGML_TYPE_SIZE[dst->type]) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  if (ggml_is_contiguous(dst)) {
    // TODO: simplify
    if (nb00 == sizeof(float)) {
      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char *dst_ptr = (char *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char *src0_ptr = (char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (type_traits[dst->type].from_float) {
        ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / GGML_BLCK_SIZE[dst->type]);
        char *dst_ptr = (char *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const float *src0_ptr = (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              quantize_row_q(src0_ptr, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        abort();  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float *dst_ptr = (float *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float *src0_ptr =
                    (float *)((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        ggml_fp16_t *dst_ptr = (ggml_fp16_t *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float *src0_ptr =
                    (float *)((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        abort();  // TODO: implement
      }
    }

    return;
  }

  // dst counters

  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == GGML_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char *src0_ptr = ((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char *dst_ptr = ((char *)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            memcpy(dst_ptr, src0_ptr, sizeof(float));

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == GGML_TYPE_F16) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char *src0_ptr = ((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char *dst_ptr = ((char *)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            *(ggml_fp16_t *)dst_ptr = *(const float *)src0_ptr;

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else {
    abort();  // TODO: implement
  }
}

static void ggml_compute_forward_dup(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_dup_f16(params, src0, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_dup_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_add

static void ggml_compute_forward_add_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(float)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

      vDSP_vadd(src0_ptr, 1, src1_ptr, 1, dst_ptr, 1, ne00);
    }
  } else {
    // src1 is not contiguous
    for (int ir = ir0; ir < ir1; ++ir) {
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int i0 = 0; i0 < ne0; i0++) {
        float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
      }
    }
  }
}

static void ggml_compute_forward_add_f16_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                             const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(float)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ggml_fp16_t *dst_ptr = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      float *src1_ptr = (float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
      }
    }
  } else {
    // src1 is not contiguous
  }
}

static void ggml_compute_forward_add_f16_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                             const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(ggml_fp16_t)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ggml_fp16_t *dst_ptr = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      ggml_fp16_t *src1_ptr = (ggml_fp16_t *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = GGML_FP16_TO_FP32(src0_ptr[i]) + GGML_FP16_TO_FP32(src1_ptr[i]);
      }
    }
  } else {
    // src1 is not contiguous
  }
}

static void ggml_compute_forward_add_q_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
  ggml_from_float_t const quantize_row_q = type_traits[type].from_float;

  // we don't support permuted src0 or src1

  // dst cannot be transposed or permuted

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float *wdata = (float *)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    // src1 and dst are same shape as src0 => same indices
    const int i13 = i03;
    const int i12 = i02;
    const int i11 = i01;

    const int i3 = i03;
    const int i2 = i02;
    const int i1 = i01;

    void *src0_row = (void *)((char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    float *src1_row = (float *)((char *)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
    void *dst_row = (void *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne00);
    // add src1
    ggml_vec_acc_f32(ne00, wdata, src1_row);
    // quantize row to dst
    quantize_row_q(wdata, dst_row, ne00);
  }
}

static void ggml_compute_forward_add(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_add_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      if (src1->type == GGML_TYPE_F16) {
        ggml_compute_forward_add_f16_f16(params, src0, src1, dst);
      } else if (src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_add_f16_f32(params, src0, src1, dst);
      } else {
      }
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_add_q_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_add1

static void ggml_compute_forward_add1_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    UNUSED(ggml_vec_add1_f32);

    vDSP_vadd((float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1, (float *)((char *)src1->data), 0,
              (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
  }
}

static void ggml_compute_forward_add1_f16_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float *)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ggml_fp16_t *dst_ptr = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = GGML_FP16_TO_FP32(src0_ptr[i]) + v;
    }
  }
}

static void ggml_compute_forward_add1_f16_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = GGML_FP16_TO_FP32(*(ggml_fp16_t *)src1->data);

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ggml_fp16_t *dst_ptr = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ggml_fp16_t *src0_ptr = (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = GGML_FP16_TO_FP32(src0_ptr[i]) + v;
    }
  }
}

static void ggml_compute_forward_add1_q_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float *)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS;

  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
  ggml_from_float_t const quantize_row_q = type_traits[type].from_float;

  // we don't support permuted src0

  // dst cannot be transposed or permuted

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float *wdata = (float *)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    void *src0_row = (void *)((char *)src0->data + (i1 * nb01 + i2 * nb02 + i3 * nb03));
    void *dst_row = (void *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne0);
    // add src1
    ggml_vec_acc1_f32(ne0, wdata, v);
    // quantize row to dst
    quantize_row_q(wdata, dst_row, ne0);
  }
}

static void ggml_compute_forward_add1(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_add1_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      if (src1->type == GGML_TYPE_F16) {
        ggml_compute_forward_add1_f16_f16(params, src0, src1, dst);
      } else if (src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_add1_f16_f32(params, src0, src1, dst);
      } else {
      }
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_add1_q_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_acc

static void ggml_compute_forward_acc_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                         struct ggml_tensor *dst) {
  // view src0 and dst with these strides and data offset inbytes during acc
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t *)opt0->data)[0];
  size_t nb2 = ((int32_t *)opt0->data)[1];
  size_t nb3 = ((int32_t *)opt0->data)[2];
  size_t offset = ((int32_t *)opt0->data)[3];
  bool inplace = (bool)((int32_t *)opt0->data)[4];

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char *)dst->data), ((char *)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src1);
  const int nc = src1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
  GGML_TENSOR_LOCALS(size_t, nb1, src1, nb);

  const size_t nb01 = nb1;
  const size_t nb02 = nb2;
  const size_t nb03 = nb3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

    vDSP_vadd((float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset), 1,
              (float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
              (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset), 1, nc);
  }
}

static void ggml_compute_forward_acc(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_acc_f32(params, src0, src1, opt0, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    default: {
    } break;
  }
}

// ggml_compute_forward_sub

static void ggml_compute_forward_sub_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      vDSP_vsub((float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
                (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1,
                (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float *dst_ptr = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float *src1_ptr = (float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
      }
    }
  }
}

static void ggml_compute_forward_sub(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sub_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_mul

static void ggml_compute_forward_mul_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }
  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  if (nb10 == sizeof(float)) {
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

      UNUSED(ggml_vec_mul_f32);

      vDSP_vmul(src0_ptr, 1, src1_ptr, 1, dst_ptr, 1, ne00);
    }
  } else {
    // src1 is not contiguous
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int64_t i0 = 0; i0 < ne00; i0++) {
        float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
      }
    }
  }
}

static void ggml_compute_forward_mul(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_mul_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_div

static void ggml_compute_forward_div_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      vDSP_vdiv((float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
                (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1,
                (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float *dst_ptr = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float *src0_ptr = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float *src1_ptr = (float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
      }
    }
  }
}

static void ggml_compute_forward_div(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_div_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_sqr

static void ggml_compute_forward_sqr_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_sqr_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sqr(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sqr_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_sqrt

static void ggml_compute_forward_sqrt_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_sqrt_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                      (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sqrt(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sqrt_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_log

static void ggml_compute_forward_log_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_log_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_log(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_log_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_sum

static void ggml_compute_forward_sum_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);
  GGML_TENSOR_LOCALS(size_t, nb0, src0, nb);

  ggml_float sum = 0;
  ggml_float row_sum = 0;

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ggml_vec_sum_ggf(ne00, &row_sum, (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
        sum += row_sum;
      }
    }
  }
  ((float *)dst->data)[0] = sum;
}

static void ggml_compute_forward_sum(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sum_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_sum_rows

static void ggml_compute_forward_sum_rows_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  for (int64_t i3 = 0; i3 < ne03; i3++) {
    for (int64_t i2 = 0; i2 < ne02; i2++) {
      for (int64_t i1 = 0; i1 < ne01; i1++) {
        float *src_row = (float *)((char *)src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float *dst_row = (float *)((char *)dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        float row_sum = 0;
        ggml_vec_sum_f32(ne00, &row_sum, src_row);
        dst_row[0] = row_sum;
      }
    }
  }
}

static void ggml_compute_forward_sum_rows(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sum_rows_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_mean

static void ggml_compute_forward_mean_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  UNUSED(ne0);
  UNUSED(ne1);
  UNUSED(ne2);
  UNUSED(ne3);

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ggml_vec_sum_f32(ne00, (float *)((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                         (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));

        *(float *)((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3) /= (float)ne00;
      }
    }
  }
}

static void ggml_compute_forward_mean(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_mean_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_argmax

static void ggml_compute_forward_argmax_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];

  const size_t nb01 = src0->nb[1];
  const size_t nb0 = dst->nb[0];

  for (int64_t i1 = 0; i1 < ne01; i1++) {
    float *src = (float *)((char *)src0->data + i1 * nb01);
    int32_t *dst_ = (int32_t *)((char *)dst->data + i1 * nb0);
    int v = 0;
    ggml_vec_argmax_f32(ne00, &v, src);
    dst_[0] = v;
  }
}

static void ggml_compute_forward_argmax(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                        struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_argmax_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne0 / ne00);
  const int nr1 = (int)(ne1 / ne01);
  const int nr2 = (int)(ne2 / ne02);
  const int nr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne03; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne02; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne01; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_vec_cpy_f32(ne00,
                                 (float *)((char *)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 +
                                           (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0),
                                 (float *)((char *)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01));
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                        struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_repeat_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_repeat_back

static void ggml_compute_forward_repeat_back_f32(const struct ggml_compute_params *params,
                                                 const struct ggml_tensor *src0, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne00 / ne0);
  const int nr1 = (int)(ne01 / ne1);
  const int nr2 = (int)(ne02 / ne2);
  const int nr3 = (int)(ne03 / ne3);

  // TODO: support for transposed / permuted tensors

  if (ggml_is_contiguous(dst)) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, dst->data, 0);
  } else {
    for (int k3 = 0; k3 < ne3; k3++) {
      for (int k2 = 0; k2 < ne2; k2++) {
        for (int k1 = 0; k1 < ne1; k1++) {
          ggml_vec_set_f32(ne0, (float *)((char *)dst->data + k1 * nb1 + k2 * nb2 + k3 * nb3), 0);
        }
      }
    }
  }

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne3; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne2; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne1; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_vec_acc_f32(ne0, (float *)((char *)dst->data + (k3)*nb3 + (k2)*nb2 + (k1)*nb1),
                                 (float *)((char *)src0->data + (i3 * ne3 + k3) * nb03 + (i2 * ne2 + k2) * nb02 +
                                           (i1 * ne1 + k1) * nb01 + (i0 * ne0) * nb00));
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                             struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_repeat_back_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_abs

static void ggml_compute_forward_abs_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_abs_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_abs(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_abs_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_sgn

static void ggml_compute_forward_sgn_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_sgn_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sgn(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sgn_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_neg

static void ggml_compute_forward_neg_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_neg_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_neg(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_neg_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_step

static void ggml_compute_forward_step_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_step_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                      (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_step(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_step_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_tanh

static void ggml_compute_forward_tanh_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_tanh_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                      (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_tanh(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_tanh_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_elu

static void ggml_compute_forward_elu_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_elu_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                     (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_elu(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_elu_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_relu

static void ggml_compute_forward_relu_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    ggml_vec_relu_f32(nc, (float *)((char *)dst->data + i * (dst->nb[1])),
                      (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_relu(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_relu_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_gelu_f32(nc, (float *)((char *)dst->data + i1 * (dst->nb[1])),
                      (float *)((char *)src0->data + i1 * (src0->nb[1])));
  }
}

static void ggml_compute_forward_gelu(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_gelu_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_gelu_quick

static void ggml_compute_forward_gelu_quick_f32(const struct ggml_compute_params *params,
                                                const struct ggml_tensor *src0, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_gelu_quick_f32(nc, (float *)((char *)dst->data + i1 * (dst->nb[1])),
                            (float *)((char *)src0->data + i1 * (src0->nb[1])));
  }
}

static void ggml_compute_forward_gelu_quick(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_gelu_quick_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_silu_f32(nc, (float *)((char *)dst->data + i1 * (dst->nb[1])),
                      (float *)((char *)src0->data + i1 * (src0->nb[1])));
  }
}

static void ggml_compute_forward_silu(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_silu_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_silu_back

static void ggml_compute_forward_silu_back_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *grad, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_silu_backward_f32(nc, (float *)((char *)dst->data + i1 * (dst->nb[1])),
                               (float *)((char *)src0->data + i1 * (src0->nb[1])),
                               (float *)((char *)grad->data + i1 * (grad->nb[1])));
  }
}

static void ggml_compute_forward_silu_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *grad, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_silu_back_f32(params, src0, grad, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_UNARY_OP_LOCALS;

  const float eps = 1e-5f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float *x = (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ggml_float)x[i00];
        }

        float mean = sum / ne00;

        float *y = (float *)((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ggml_float sum2 = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          float v = x[i00] - mean;
          y[i00] = v;
          sum2 += (ggml_float)(v * v);
        }

        float variance = sum2 / ne00;
        const float scale = 1.0f / sqrtf(variance + eps);

        ggml_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ggml_compute_forward_norm(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_norm_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

static void ggml_compute_forward_rms_norm_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_UNARY_OP_LOCALS;

  const float eps = 1e-6f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float *x = (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ggml_float)(x[i00] * x[i00]);
        }

        const float mean = sum / ne00;

        float *y = (float *)((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        memcpy(y, x, ne00 * sizeof(float));

        const float scale = 1.0f / sqrtf(mean + eps);

        ggml_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ggml_compute_forward_rms_norm(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_rms_norm_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

static void ggml_compute_forward_rms_norm_back_f32(const struct ggml_compute_params *params,
                                                   const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                   struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_BINARY_OP_LOCALS;

  const float eps = 1e-6f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        // src1 is same shape as src0 => same indices
        const int64_t i11 = i01;
        const int64_t i12 = i02;
        const int64_t i13 = i03;

        const float *x = (float *)((char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        const float *dz = (float *)((char *)src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);

        ggml_float sum_xx = 0.0;
        ggml_float sum_xdz = 0.0;

        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum_xx += (ggml_float)(x[i00] * x[i00]);
          sum_xdz += (ggml_float)(x[i00] * dz[i00]);
        }

        const float mean_eps = (float)(sum_xx) / ne00 + eps;
        const float sum_eps = (float)(sum_xx) + eps * ne00;
        //  we could cache rms from forward pass to improve performance.
        //  to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
        const float rrms = 1.0f / sqrtf(mean_eps);

        {
          // z = rms_norm(x)
          //
          // rms_norm(src0) =
          //     scale(
          //         src0,
          //         div(
          //             1,
          //             sqrt(
          //                 add(
          //                     scale(
          //                         sum(
          //                             sqr(
          //                                 src0)),
          //                         (1.0/N)),
          //                     eps))));

          // postorder:
          // ## op    args         grad
          // 00 param src0         grad[#00]
          // 01 const 1
          // 02 sqr   (#00)        grad[#02]
          // 03 sum   (#02)        grad[#03]
          // 04 const 1/N
          // 05 scale (#03, #04)   grad[#05]
          // 06 const eps
          // 07 add   (#05, #06)   grad[#07]
          // 08 sqrt  (#07)        grad[#08]
          // 09 div   (#01,#08)    grad[#09]
          // 10 scale (#00,#09)    grad[#10]
          //
          // backward pass, given grad[#10]
          // #10: scale
          // grad[#00] += scale(grad[#10],#09)
          // grad[#09] += sum(mul(grad[#10],#00))
          // #09: div
          // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
          // #08: sqrt
          // grad[#07] += mul(grad[#08], div(0.5, #08))
          // #07: add
          // grad[#05] += grad[#07]
          // #05: scale
          // grad[#03] += scale(grad[#05],#04)
          // #03: sum
          // grad[#02] += repeat(grad[#03], #02)
          // #02:
          // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
          //
          // substitute and simplify:
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#02] = repeat(grad[#03], #02)
          // grad[#02] = repeat(scale(grad[#05],#04), #02)
          // grad[#02] = repeat(scale(grad[#07],#04), #02)
          // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5,
          // #08) * (1/N)), #02)), 2.0) grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) *
          // div(1,#07) * div(0.5, #08) * (1/N))), 2.0) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N))) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps))) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps)) grad[#00] = scale(dz, rrms) + scale(x,
          // sum(mul(dz,x)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(dz, rrms) + scale(x, sum_xdz *
          // div(-1,rms*N*mean_eps)) a = b*c + d*e a = b*c*f/f + d*e*f/f a = (b*c*f + d*e*f)*(1/f) a = (b*c*(1/c) +
          // d*e*(1/c))*(1/(1/c)) a = (b + d*e/c)*c b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps) a =
          // (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
          // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
          // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
          // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
          // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
          // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
          // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
          // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        }
        // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        // post-order:
        // dx := x
        // dx := scale(dx,-mean_xdz/mean_eps)
        // dx := add(dx, dz)
        // dx := scale(dx, rrms)
        float *dx = (float *)((char *)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ggml_vec_cpy_f32(ne00, dx, x);
        // ggml_vec_scale_f32(ne00, dx, -mean_xdz/mean_eps);
        ggml_vec_scale_f32(ne00, dx, (float)(-sum_xdz) / sum_eps);
        ggml_vec_acc_f32(ne00, dx, dz);
        ggml_vec_scale_f32(ne00, dx, rrms);
      }
    }
  }
}

static void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_rms_norm_back_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_mul_mat

// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                  struct ggml_tensor *dst) {
  // const int64_t ne00 = src0->ne[0];
  // const int64_t ne01 = src0->ne[1];

  const int64_t ne10 = src1->ne[0];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];

  // TODO: find the optimal values for these
  if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {
    /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
    return true;
  }

  return false;
}

static void ggml_compute_forward_mul_mat(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ggml_type type = src0->type;

  ggml_vec_dot_t const vec_dot = type_traits[type].vec_dot;
  enum ggml_type const vec_dot_type = type_traits[type].vec_dot_type;
  ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;

  // we don't support permuted src0 or src1

  // dst cannot be transposed or permuted

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
    // TODO: handle case when src0 is broadcast-able into src1 across 2nd,3rd dimension
    //       ref: https://github.com/ggerganov/ggml/pull/224

    if (params->ith != 0) {
      return;
    }

    if (params->type == GGML_TASK_INIT) {
      return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
      return;
    }

    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        const void *x = (char *)src0->data + i03 * nb03 + i02 * nb02;
        const float *y = (float *)((char *)src1->data + i02 * nb12 + i03 * nb13);

        float *d = (float *)((char *)dst->data + i02 * nb2 + i03 * nb3);

        if (type != GGML_TYPE_F32) {
          float *const wdata = params->wdata;
          ggml_to_float_t const to_float = type_traits[type].to_float;

          size_t id = 0;
          for (int64_t i01 = 0; i01 < ne01; ++i01) {
            to_float((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01, wdata + id, ne00);
            id += ne00;
          }

          x = wdata;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ne11, ne01, ne10, 1.0f, y, ne10, x, ne00, 0.0f, d, ne01);
      }
    }

    // printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

    return;
  }

  if (params->type == GGML_TASK_INIT) {
    if (src1->type != vec_dot_type) {
      char *wdata = params->wdata;
      const size_t row_size = ne10 * GGML_TYPE_SIZE[vec_dot_type] / GGML_BLCK_SIZE[vec_dot_type];

      for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
          for (int64_t i11 = 0; i11 < ne11; ++i11) {
            from_float_to_vec_dot((float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11), (void *)wdata,
                                  ne10);
            wdata += row_size;
          }
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by src0 rows
  const int64_t dr = (ne01 + nth - 1) / nth;

  const int64_t ir10 = dr * ith;
  const int64_t ir11 = MIN(ir10 + dr, ne01);

  // src1 rows
  const int64_t nr1 = ne11 * ne12 * ne13;

  void *wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
  const size_t row_size = ne10 * GGML_TYPE_SIZE[vec_dot_type] / GGML_BLCK_SIZE[vec_dot_type];

  for (int64_t ir1 = 0; ir1 < nr1; ++ir1) {
    const int64_t i13 = (ir1 / (ne12 * ne11));
    const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
    const int64_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

    const int64_t ir0 = (ir1 / ne11) % (ne02 * ne03);
    const int64_t i03 = (ir0 / (ne02));
    // Hack for "Falcon multi-query-attention key stutter" / alternative to ggml_repeat2.
    // See https://github.com/ggerganov/llama.cpp/issues/1602#issuecomment-1606087470:
    // GG: this is likely the correct way to broadcast, though need some more thought
    //     therefore leaving the comments to remind us for now
    const int64_t i02 = (i12 / (ne12 / ne02));
    // Original from PR/224 (and also essential/correct for non-broadcast matmuls in Falcon)
    // const int64_t i02 = (ir0 - i03*ne02);

    const int64_t i1 = i11;
    const int64_t i2 = i12;
    const int64_t i3 = i13;

    const char *src0_row = (const char *)src0->data + (0 + i02 * nb02 + i03 * nb03);
    const char *src1_col = (const char *)wdata + (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size;

    float *dst_col = (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    for (int64_t ir = ir10; ir < ir11; ++ir) {
      vec_dot(ne00, &dst_col[ir], src0_row + ir * nb01, src1_col);
    }
  }
}

// ggml_compute_forward_out_prod

static void ggml_compute_forward_out_prod_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  if (params->type == GGML_TASK_INIT) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, dst->data, 0);
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by last three dimensions

  // total rows in dst
  const int64_t nr = ne1 * ne2 * ne3;

  // rows per thread
  const int64_t dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = MIN(ir0 + dr, nr);

  // dst[:,:,:,:] = 0
  // for i2,i3:
  //   for i1:
  //     for i01:
  //       for i0:
  //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

  for (int64_t ir = ir0; ir < ir1; ++ir) {
    // dst indices
    const int64_t i3 = ir / (ne2 * ne1);
    const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int64_t i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    const int64_t i02 = i2;
    const int64_t i03 = i3;

    // const int64_t i10 = i1;
    const int64_t i12 = i2;
    const int64_t i13 = i3;

    for (int64_t i01 = 0; i01 < ne01; ++i01) {
      const int64_t i11 = i01;

      float *s0 = (float *)((char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
      float *s1 = (float *)((char *)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
      float *d = (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

      ggml_vec_mad_f32(ne0, d, s0, *s1);
    }
  }
}

static void ggml_compute_forward_out_prod(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_F16: {
      abort();  // todo
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_out_prod_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scale factor
  const float v = *(float *)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const size_t nb01 = src0->nb[1];

  const size_t nb1 = dst->nb[1];

  for (int i1 = ir0; i1 < ir1; i1++) {
    if (dst->data != src0->data) {
      // src0 is same shape as dst => same indices
      memcpy((char *)dst->data + i1 * nb1, (char *)src0->data + i1 * nb01, nc * sizeof(float));
    }
    ggml_vec_scale_f32(nc, (float *)((char *)dst->data + i1 * nb1), v);
  }
}

static void ggml_compute_forward_scale(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                       const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_scale_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_set

static void ggml_compute_forward_set_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                         struct ggml_tensor *dst) {
  // view src0 and dst with these strides and data offset inbytes during set
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t *)opt0->data)[0];
  size_t nb2 = ((int32_t *)opt0->data)[1];
  size_t nb3 = ((int32_t *)opt0->data)[2];
  size_t offset = ((int32_t *)opt0->data)[3];
  bool inplace = (bool)((int32_t *)opt0->data)[4];

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char *)dst->data), ((char *)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src1);
  const int nc = src1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
  GGML_TENSOR_LOCALS(size_t, nb1, src1, nb);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

    ggml_vec_cpy_f32(nc, (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
                     (float *)((char *)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
  }
}

static void ggml_compute_forward_set(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                     struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_set_f32(params, src0, src1, opt0, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    default: {
    } break;
  }
}

// ggml_compute_forward_cpy

static void ggml_compute_forward_cpy(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                     struct ggml_tensor *dst) {
  ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_cont

static void ggml_compute_forward_cont(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_reshape

static void ggml_compute_forward_reshape(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         struct ggml_tensor *dst) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
  UNUSED(dst);
}

// ggml_compute_forward_view

static void ggml_compute_forward_view(const struct ggml_compute_params *params, const struct ggml_tensor *src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ggml_compute_forward_permute

static void ggml_compute_forward_permute(const struct ggml_compute_params *params, const struct ggml_tensor *src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ggml_compute_forward_transpose

static void ggml_compute_forward_transpose(const struct ggml_compute_params *params, const struct ggml_tensor *src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ggml_compute_forward_get_rows

static void ggml_compute_forward_get_rows_q(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);
  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t *)src1->data)[i];

    dequantize_row_q((const void *)((char *)src0->data + r * src0->nb[1]),
                     (float *)((char *)dst->data + i * dst->nb[1]), nc);
  }
}

static void ggml_compute_forward_get_rows_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t *)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ggml_fp16_t v = ((ggml_fp16_t *)((char *)src0->data + r * src0->nb[1]))[j];
      ((float *)((char *)dst->data + i * dst->nb[1]))[j] = GGML_FP16_TO_FP32(v);
    }
  }
}

static void ggml_compute_forward_get_rows_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t *)src1->data)[i];

    ggml_vec_cpy_f32(nc, (float *)((char *)dst->data + i * dst->nb[1]),
                     (float *)((char *)src0->data + r * src0->nb[1]));
  }
}

static void ggml_compute_forward_get_rows(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_get_rows_q(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      ggml_compute_forward_get_rows_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_get_rows_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_get_rows_back

static void ggml_compute_forward_get_rows_back_f32_f16(const struct ggml_compute_params *params,
                                                       const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                       const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  ggml_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t *)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ggml_fp16_t v = ((ggml_fp16_t *)((char *)src0->data + i * src0->nb[1]))[j];
      ((float *)((char *)dst->data + r * dst->nb[1]))[j] += GGML_FP16_TO_FP32(v);
    }
  }
}

static void ggml_compute_forward_get_rows_back_f32(const struct ggml_compute_params *params,
                                                   const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                   struct ggml_tensor *dst) {
  // ggml_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == GGML_TASK_INIT) {
    memset(dst->data, 0, ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t *)src1->data)[i];

    ggml_vec_add_f32(nc, (float *)((char *)dst->data + r * dst->nb[1]), (float *)((char *)dst->data + r * dst->nb[1]),
                     (float *)((char *)src0->data + i * src0->nb[1]));
  }
}

static void ggml_compute_forward_get_rows_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                               struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_get_rows_back_f32_f16(params, src0, src1, opt0, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_get_rows_back_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_diag

static void ggml_compute_forward_diag_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  GGML_TENSOR_UNARY_OP_LOCALS;

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = 0; i2 < ne2; i2++) {
      for (int i1 = 0; i1 < ne1; i1++) {
        float *d = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
        float *s = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02);
        for (int i0 = 0; i0 < i1; i0++) {
          d[i0] = 0;
        }
        d[i1] = s[i1];
        for (int i0 = i1 + 1; i0 < ne0; i0++) {
          d[i0] = 0;
        }
      }
    }
  }
}

static void ggml_compute_forward_diag(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_diag_mask_inf

static void ggml_compute_forward_diag_mask_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst,
                                               const float value) {
  const int ith = params->ith;
  const int nth = params->nth;

  const int n_past = ((int32_t *)src1->data)[0];
  const bool inplace = (bool)((int32_t *)src1->data)[1];

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase

    memcpy(((char *)dst->data), ((char *)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];
  const int nr = src0->ne[1];
  const int nz = n / nr;

  for (int k = 0; k < nz; k++) {
    for (int j = ith; j < nr; j += nth) {
      for (int i = n_past; i < nc; i++) {
        if (i > n_past + j) {
          *(float *)((char *)dst->data + k * dst->nb[2] + j * dst->nb[1] + i * dst->nb[0]) = value;
        }
      }
    }
  }
}

static void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_mask_f32(params, src0, src1, dst, -INFINITY);
    } break;
    default: {
    } break;
  }
}

static void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params *params,
                                                const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_mask_f32(params, src0, src1, dst, 0);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *sp = (float *)((char *)src0->data + i1 * src0->nb[1]);
    float *dp = (float *)((char *)dst->data + i1 * dst->nb[1]);

    float max = -INFINITY;
    ggml_vec_max_f32(nc, &max, sp);

    ggml_float sum = 0.0;

    uint16_t scvt;
    for (int i = 0; i < nc; i++) {
      if (sp[i] == -INFINITY) {
        dp[i] = 0.0f;
      } else {
        // const float val = (sp[i] == -INFINITY) ? 0.0 : exp(sp[i] - max);
        ggml_fp16_t s = sp[i] - max;
        memcpy(&scvt, &s, sizeof(scvt));
        const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
        sum += (ggml_float)val;
        dp[i] = val;
      }
    }

    sum = 1.0 / sum;
    ggml_vec_scale_f32(nc, dp, sum);
  }
}

static void ggml_compute_forward_soft_max(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_soft_max_f32(params, src0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_soft_max_back

static void ggml_compute_forward_soft_max_back_f32(const struct ggml_compute_params *params,
                                                   const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                   struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *dy = (float *)((char *)src0->data + i1 * src0->nb[1]);
    float *y = (float *)((char *)src1->data + i1 * src1->nb[1]);
    float *dx = (float *)((char *)dst->data + i1 * dst->nb[1]);

    // Jii = yi - yi*yi
    // Jij = -yi*yj
    // J = diag(y)-y.T*y
    // dx = J * dy
    // dxk = sum_i(Jki * dyi)
    // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
    // dxk = sum_i(-yk*yi * dyi) + yk*dyk
    // dxk = -yk * sum_i(yi * dyi) + yk*dyk
    // dxk = -yk * dot(y, dy) + yk*dyk
    // dxk = yk * (- dot(y, dy) + dyk)
    // dxk = yk * (dyk - dot(y, dy))
    //
    // post-order:
    // dot_y_dy := dot(y, dy)
    // dx := dy
    // dx := dx - dot_y_dy
    // dx := dx * y

    // linear runtime, no additional memory
    float dot_y_dy = 0;
    ggml_vec_dot_f32(nc, &dot_y_dy, y, dy);
    ggml_vec_cpy_f32(nc, dx, dy);
    ggml_vec_acc1_f32(nc, dx, -dot_y_dy);
    ggml_vec_mul_f32(nc, dx, dx, y);
  }
}

static void ggml_compute_forward_soft_max_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_soft_max_back_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_alibi

static void ggml_compute_forward_alibi_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_head = ((int32_t *)src1->data)[1];
  const float max_bias = ((float *)src1->data)[2];

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past

  const int n = ggml_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const int nb0 = src0->nb[0];
  const int nb1 = src0->nb[1];
  const int nb2 = src0->nb[2];

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        float *const src = (float *)((char *)src0->data + i * nb0 + j * nb1 + k * nb2);
        float *pdst = (float *)((char *)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        pdst[0] = i * m_k + src[0];
      }
    }
  }
}

static void ggml_compute_forward_alibi_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_head = ((int32_t *)src1->data)[1];
  const float max_bias = ((float *)src1->data)[2];

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past

  const int n = ggml_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const int nb0 = src0->nb[0];
  const int nb1 = src0->nb[1];
  const int nb2 = src0->nb[2];

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        ggml_fp16_t *const src = (ggml_fp16_t *)((char *)src0->data + i * nb0 + j * nb1 + k * nb2);
        float *pdst = (float *)((char *)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        // we return F32
        pdst[0] = i * m_k + GGML_FP16_TO_FP32(src[0]);
      }
    }
  }
}

static void ggml_compute_forward_alibi(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                       const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_alibi_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_alibi_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_COUNT: {
    } break;
  }
}

// ggml_compute_forward_clamp

static void ggml_compute_forward_clamp_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const float min = ((float *)src1->data)[0];
  const float max = ((float *)src1->data)[1];

  const int ith = params->ith;
  const int nth = params->nth;

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  const size_t nb01 = src0->nb[1];

  const size_t nb1 = dst->nb[1];

  for (int j = ith; j < n; j += nth) {
    float *dst_ptr = (float *)((char *)dst->data + j * nb1);
    float *src0_ptr = (float *)((char *)src0->data + j * nb01);

    for (int i = 0; i < nc; i++) {
      dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
    }
  }
}

static void ggml_compute_forward_clamp(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                       const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_clamp_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_COUNT: {
    } break;
  }
}

// ggml_compute_forward_rope

static void ggml_compute_forward_rope_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t *)src1->data)[0];
  const int n_dims = ((int32_t *)src1->data)[1];
  const int mode = ((int32_t *)src1->data)[2];
  const int n_ctx = ((int32_t *)src1->data)[3];

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;
  const bool is_glm = mode & 4;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) {
          continue;
        }
        if (ir > ir1) {
          break;
        }

        float theta = (float)p;

        if (is_glm) {
          theta = MIN(p, n_ctx - 2);
          float block_theta = MAX(p - (n_ctx - 2), 0);
          for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            const float cos_block_theta = cosf(block_theta);
            const float sin_block_theta = sinf(block_theta);

            theta *= theta_scale;
            block_theta *= theta_scale;

            const float *const src = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float *dst_data = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims / 2];
            const float x2 = src[n_dims];
            const float x3 = src[n_dims / 2 * 3];

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            dst_data[n_dims] = x2 * cos_block_theta - x3 * sin_block_theta;
            dst_data[n_dims / 2 * 3] = x2 * sin_block_theta + x3 * cos_block_theta;
          }
        } else if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const float *const src = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float *dst_data = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[1] = x0 * sin_theta + x1 * cos_theta;
          }
        } else {
          // TODO: this is probably wrong, but I can't figure it out ..
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float *const src = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float *dst_data = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = src[0];
              const float x1 = src[n_dims / 2];

              dst_data[0] = x0 * cos_theta - x1 * sin_theta;
              dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t *)src1->data)[0];
  const int n_dims = ((int32_t *)src1->data)[1];
  const int mode = ((int32_t *)src1->data)[2];
  const int n_ctx = ((int32_t *)src1->data)[3];

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;
  const bool is_glm = mode & 4;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) {
          continue;
        }
        if (ir > ir1) {
          break;
        }

        float theta = (float)p;

        if (is_glm) {
          theta = MIN(p, n_ctx - 2);
          float block_theta = MAX(p - (n_ctx - 2), 0);
          for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            const float cos_block_theta = cosf(block_theta);
            const float sin_block_theta = sinf(block_theta);

            theta *= theta_scale;
            block_theta *= theta_scale;

            const ggml_fp16_t *const src =
                (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t *dst_data = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = GGML_FP16_TO_FP32(src[0]);
            const float x1 = GGML_FP16_TO_FP32(src[n_dims / 2]);
            const float x2 = GGML_FP16_TO_FP32(src[n_dims]);
            const float x3 = GGML_FP16_TO_FP32(src[n_dims / 2 * 3]);

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            dst_data[n_dims] = x2 * cos_block_theta - x3 * sin_block_theta;
            dst_data[n_dims / 2 * 3] = x2 * sin_block_theta + x3 * cos_block_theta;
          }
        }
        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ggml_fp16_t *const src =
                (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t *dst_data = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = GGML_FP16_TO_FP32(src[0]);
            const float x1 = GGML_FP16_TO_FP32(src[1]);

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[1] = x0 * sin_theta + x1 * cos_theta;
          }
        } else {
          // TODO: this is probably wrong, but I can't figure it out ..
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ggml_fp16_t *const src =
                  (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ggml_fp16_t *dst_data = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = GGML_FP16_TO_FP32(src[0]);
              const float x1 = GGML_FP16_TO_FP32(src[n_dims / 2]);

              dst_data[0] = x0 * cos_theta - x1 * sin_theta;
              dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                      const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_rope_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_rope_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_rope_back

static void ggml_compute_forward_rope_back_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t *)src1->data)[0];
  const int n_dims = ((int32_t *)src1->data)[1];
  const int mode = ((int32_t *)src1->data)[2];

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) {
          continue;
        }
        if (ir > ir1) {
          break;
        }

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const float *const dy = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float *dx = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = dy[0];
            const float dy1 = dy[1];

            dx[0] = dy0 * cos_theta + dy1 * sin_theta;
            dx[1] = -dy0 * sin_theta + dy1 * cos_theta;
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float *const dy = (float *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float *dx = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = dy[0];
              const float dy1 = dy[n_dims / 2];

              dx[0] = dy0 * cos_theta + dy1 * sin_theta;
              dx[n_dims / 2] = -dy0 * sin_theta + dy1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_back_f16(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // y = rope(x, src1)
  // dx = rope_back(dy, src1)
  // src0 is dy, src1 contains options

  const int n_past = ((int32_t *)src1->data)[0];
  const int n_dims = ((int32_t *)src1->data)[1];
  const int mode = ((int32_t *)src1->data)[2];

  GGML_TENSOR_UNARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) {
          continue;
        }
        if (ir > ir1) {
          break;
        }

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ggml_fp16_t *const dy =
                (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t *dx = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = GGML_FP16_TO_FP32(dy[0]);
            const float dy1 = GGML_FP16_TO_FP32(dy[1]);

            dx[0] = dy0 * cos_theta + dy1 * sin_theta;
            dx[1] = -dy0 * sin_theta + dy1 * cos_theta;
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ggml_fp16_t *const dy =
                  (ggml_fp16_t *)((char *)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ggml_fp16_t *dx = (ggml_fp16_t *)((char *)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = GGML_FP16_TO_FP32(dy[0]);
              const float dy1 = GGML_FP16_TO_FP32(dy[n_dims / 2]);

              dx[0] = dy0 * cos_theta + dy1 * sin_theta;
              dx[n_dims / 2] = -dy0 * sin_theta + dy1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_back(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_rope_back_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_rope_back_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_conv_1d

static void ggml_compute_forward_conv_1d_s1_ph_f16_f32(const struct ggml_compute_params *params,
                                                       const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                       struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ggml_fp16_t *const src = (ggml_fp16_t *)((char *)src0->data + i02 * nb02 + i01 * nb01);
          ggml_fp16_t *dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float *const src = (float *)((char *)src1->data + i11 * nb11);
        ggml_fp16_t *dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *dst_data = (float *)((char *)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f16(ew0, &v, (ggml_fp16_t *)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (ggml_fp16_t *)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s1_ph_f32(const struct ggml_compute_params *params,
                                                   const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                   struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float *const wdata = (float *)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float *const src = (float *)((char *)src0->data + i02 * nb02 + i01 * nb01);
          float *dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float *const wdata = (float *)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float *const src = (float *)((char *)src1->data + i11 * nb11);
        float *dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *dst_data = (float *)((char *)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f32(ew0, &v, (float *)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (float *)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s1_ph(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_1d_s1_ph_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_conv_1d_s1_ph_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

static void ggml_compute_forward_conv_1d_s2_ph_f16_f32(const struct ggml_compute_params *params,
                                                       const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                       struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ggml_fp16_t *const src = (ggml_fp16_t *)((char *)src0->data + i02 * nb02 + i01 * nb01);
          ggml_fp16_t *dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float *const src = (float *)((char *)src1->data + i11 * nb11);
        ggml_fp16_t *dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *dst_data = (float *)((char *)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f16(ew0, &v, (ggml_fp16_t *)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (ggml_fp16_t *)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s2_ph_f32(const struct ggml_compute_params *params,
                                                   const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                   struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float *const wdata = (float *)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float *const src = (float *)((char *)src0->data + i02 * nb02 + i01 * nb01);
          float *dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float *const wdata = (float *)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float *const src = (float *)((char *)src1->data + i11 * nb11);
        float *dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *dst_data = (float *)((char *)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f32(ew0, &v, (float *)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (float *)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s2_ph(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_1d_s2_ph_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_conv_1d_s2_ph_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_conv_1d

static void ggml_compute_forward_conv_1d(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                         struct ggml_tensor *dst) {
  const int32_t s0 = ((const int32_t *)(opt0->data))[0];
  if (s0 == 1) {
    ggml_compute_forward_conv_1d_s1_ph(params, src0, src1, dst);
  } else if (s0 == 2) {
    ggml_compute_forward_conv_1d_s2_ph(params, src0, src1, dst);
  } else {
    abort();  // only stride 1 and 2 supported
  };
}

// ggml_compute_forward_conv_2d_sk_p0

static void ggml_compute_forward_conv_2d_sk_p0_f16_f32(const struct ggml_compute_params *params,
                                                       const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                       struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk0 = ne00;
  const int nk1 = ne01;

  // size of the convolution row - the kernel size unrolled across all channels
  const int ew0 = nk0 * nk1 * ne02;

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare source data (src1)
    {
      ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + 0;

      for (int i13 = 0; i13 < ne13; i13++) {
        for (int i12 = 0; i12 < ne12; i12++) {
          const float *const src = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12);
          ggml_fp16_t *dst_data = wdata + i13 * (ne1 * ne0 * ew0);

          for (int i1 = 0; i1 < ne1; i1++) {
            for (int i0 = 0; i0 < ne0; i0++) {
              for (int ik1 = 0; ik1 < nk1; ik1++) {
                for (int ik0 = 0; ik0 < nk0; ik0++) {
                  dst_data[(i1 * ne0 + i0) * ew0 + i12 * (nk0 * nk1) + ik1 * nk0 + ik0] =
                      src[(i1 * nk1 + ik1) * ne10 + (i0 * nk0 + ik0)];
                }
              }
            }
          }
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total patches in dst
  const int np = ne2;

  // patches per thread
  const int dp = (np + nth - 1) / nth;

  // patch range for this thread
  const int ip0 = dp * ith;
  const int ip1 = MIN(ip0 + dp, np);

  ggml_fp16_t *const wdata = (ggml_fp16_t *)params->wdata + 0;

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = ip0; i2 < ip1; i2++) {
      float *dst_data = (float *)((char *)dst->data + i3 * nb3 + i2 * nb2);

      for (int i1 = 0; i1 < ne1; ++i1) {
        for (int i0 = 0; i0 < ne0; ++i0) {
          ggml_vec_dot_f16(ew0, dst_data + i1 * ne0 + i0, (ggml_fp16_t *)((char *)src0->data + i2 * nb03),
                           (ggml_fp16_t *)wdata + i3 * nb3 + (i1 * ne0 + i0) * ew0);
        }
      }
    }
  }
}

static void ggml_compute_forward_conv_2d_sk_p0(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               const struct ggml_tensor *src1, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_2d_sk_p0_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32:
    default: {
    } break;
  }
}

// ggml_compute_forward_conv_2d

static void ggml_compute_forward_conv_2d(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1, const struct ggml_tensor *opt0,
                                         struct ggml_tensor *dst) {
  const int32_t s0 = ((const int32_t *)(opt0->data))[0];
  const int32_t s1 = ((const int32_t *)(opt0->data))[1];

  if (s0 == src0->ne[0] && s1 == src0->ne[1]) {
    ggml_compute_forward_conv_2d_sk_p0(params, src0, src1, dst);
  } else {
    abort();  // only stride equal to kernel size is supported
  }
}

// ggml_compute_forward_pool_1d_sk_p0

static void ggml_compute_forward_pool_1d_sk_p0(const struct ggml_compute_params *params, const enum ggml_op_pool op,
                                               const struct ggml_tensor *src, const int k, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const char *cdata = (const char *)src->data;
  const char *const data_end = cdata + ggml_nbytes(src);
  float *drow = (float *)dst->data;

  const int64_t rs = dst->ne[0];

  while (cdata < data_end) {
    const float *const srow = (const float *)cdata;

    int j = 0;

    for (int64_t i = 0; i < rs; ++i) {
      switch (op) {
        case GGML_OP_POOL_AVG:
          drow[i] = 0;
          break;
        case GGML_OP_POOL_MAX:
          drow[i] = -FLT_MAX;
          break;
        case GGML_OP_POOL_COUNT:

          break;
      }
      for (int ki = 0; ki < k; ++ki) {
        switch (op) {
          case GGML_OP_POOL_AVG:
            drow[i] += srow[j];
            break;
          case GGML_OP_POOL_MAX:
            if (srow[j] > drow[i]) {
              drow[i] = srow[j];
            }
            break;
          case GGML_OP_POOL_COUNT:

            break;
        }
        ++j;
      }
      switch (op) {
        case GGML_OP_POOL_AVG:
          drow[i] /= k;
          break;
        case GGML_OP_POOL_MAX:
        case GGML_OP_POOL_COUNT:
          break;
      }
    }

    cdata += src->nb[1];
    drow += rs;
  }
}

// ggml_compute_forward_pool_1d

static void ggml_compute_forward_pool_1d(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  const int *opts = (const int *)opt0->data;
  enum ggml_op_pool op = opts[0];
  const int k0 = opts[1];

  ggml_compute_forward_pool_1d_sk_p0(params, op, src0, k0, dst);
}

// ggml_compute_forward_pool_2d_sk_p0

static void ggml_compute_forward_pool_2d_sk_p0(const struct ggml_compute_params *params, const enum ggml_op_pool op,
                                               const struct ggml_tensor *src, const int k0, const int k1,
                                               struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const char *cdata = (const char *)src->data;
  const char *const data_end = cdata + ggml_nbytes(src);

  const int64_t px = dst->ne[0];
  const int64_t py = dst->ne[1];
  const int64_t pa = px * py;

  float *dplane = (float *)dst->data;

  const int ka = k0 * k1;

  while (cdata < data_end) {
    for (int oy = 0; oy < py; ++oy) {
      float *const drow = dplane + oy * px;
      for (int ox = 0; ox < px; ++ox) {
        float *const out = drow + ox;
        switch (op) {
          case GGML_OP_POOL_AVG:
            *out = 0;
            break;
          case GGML_OP_POOL_MAX:
            *out = -FLT_MAX;
            break;
          case GGML_OP_POOL_COUNT:

            break;
        }

        const int ix = ox * k0;
        const int iy = oy * k1;

        for (int ky = 0; ky < k1; ++ky) {
          const float *const srow = (const float *)(cdata + src->nb[1] * (iy + ky));
          for (int kx = 0; kx < k0; ++kx) {
            int j = ix + kx;
            switch (op) {
              case GGML_OP_POOL_AVG:
                *out += srow[j];
                break;
              case GGML_OP_POOL_MAX:
                if (srow[j] > *out) {
                  *out = srow[j];
                }
                break;
              case GGML_OP_POOL_COUNT:

                break;
            }
          }
        }
        switch (op) {
          case GGML_OP_POOL_AVG:
            *out /= ka;
            break;
          case GGML_OP_POOL_MAX:
          case GGML_OP_POOL_COUNT:
            break;
        }
      }
    }

    cdata += src->nb[2];
    dplane += pa;
  }
}

// ggml_compute_forward_pool_2d

static void ggml_compute_forward_pool_2d(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                         const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  const int *opts = (const int *)opt0->data;
  enum ggml_op_pool op = opts[0];
  const int k0 = opts[1];
  const int k1 = opts[2];

  ggml_compute_forward_pool_2d_sk_p0(params, op, src0, k0, k1, dst);
}

// ggml_compute_forward_flash_attn

static void ggml_compute_forward_flash_attn_f32(const struct ggml_compute_params *params, const struct ggml_tensor *q,
                                                const struct ggml_tensor *k, const struct ggml_tensor *v,
                                                const bool masked, struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb);
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb);
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne);
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

  // dst cannot be transposed or permuted

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float *S = (float *)params->wdata + ith * (Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    for (int64_t ic = 0; ic < nek1; ++ic) {
      // k indices
      const int ik3 = iq3;
      const int ik2 = iq2;
      const int ik1 = ic;

      // S indices
      const int i1 = ik1;

      ggml_vec_dot_f32(neq0, S + i1, (float *)((char *)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                       (float *)((char *)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
    }

    // scale
    ggml_vec_scale_f32(nek1, S, scale);

    if (masked) {
      for (int64_t i = P; i < M; i++) {
        if (i > P + iq1) {
          S[i] = -INFINITY;
        }
      }
    }

    // softmax
    {
      float max = -INFINITY;
      ggml_vec_max_f32(M, &max, S);

      ggml_float sum = 0.0;
      {
#ifdef GGML_SOFT_MAX_ACCELERATE
        max = -max;
        vDSP_vsadd(S, 1, &max, S, 1, Mup);
        vvexpf(S, S, &Mup);
        ggml_vec_sum_f32(Mup, &sum, S);
#else
        uint16_t scvt[GGML_SOFT_MAX_UNROLL];
        ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
          float *SS = S + i;

          for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
            if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
              ggml_fp16_t s = SS[j] - max;
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
              sump[j] += (ggml_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
#endif
      }

      sum = 1.0 / sum;
      ggml_vec_scale_f32(M, S, sum);
    }

    for (int64_t ic = 0; ic < nev1; ++ic) {
      // dst indices
      const int i1 = iq1;
      const int i2 = iq2;
      const int i3 = iq3;

      ggml_vec_dot_f32(nek1, (float *)((char *)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (float *)((char *)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S);
    }
  }
}

static void ggml_compute_forward_flash_attn_f16(const struct ggml_compute_params *params, const struct ggml_tensor *q,
                                                const struct ggml_tensor *k, const struct ggml_tensor *v,
                                                const bool masked, struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb);
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb);
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne);
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

  // dst cannot be transposed or permuted

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float *S = (float *)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    if (GGML_VEC_DOT_UNROLL > 2 || nek1 % GGML_VEC_DOT_UNROLL != 0) {
      for (int64_t ic = 0; ic < nek1; ++ic) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ggml_vec_dot_f16(neq0, S + i1, (ggml_fp16_t *)((char *)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                         (ggml_fp16_t *)((char *)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    } else {
      for (int64_t ic = 0; ic < nek1; ic += GGML_VEC_DOT_UNROLL) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ggml_vec_dot_f16_unroll(neq0, nbk1, S + i1, ((char *)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                                (ggml_fp16_t *)((char *)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    }

    // scale
    ggml_vec_scale_f32(nek1, S, scale);

    if (masked) {
      for (int64_t i = P; i < M; i++) {
        if (i > P + iq1) {
          S[i] = -INFINITY;
        }
      }
    }

    // softmax
    {
      float max = -INFINITY;
      ggml_vec_max_f32(M, &max, S);

      ggml_float sum = 0.0;
      {
#ifdef GGML_SOFT_MAX_ACCELERATE
        max = -max;
        vDSP_vsadd(S, 1, &max, S, 1, Mup);
        vvexpf(S, S, &Mup);
        ggml_vec_sum_f32(Mup, &sum, S);
#else
        uint16_t scvt[GGML_SOFT_MAX_UNROLL];
        ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
          float *SS = S + i;

          for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
            if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
              ggml_fp16_t s = SS[j] - max;
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
              sump[j] += (ggml_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
#endif
      }

      sum = 1.0 / sum;
      ggml_vec_scale_f32(M, S, sum);
    }

    ggml_fp16_t *S16 = (ggml_fp16_t *)((float *)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32) + Mup);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = S[i];
    }

    if (GGML_VEC_DOT_UNROLL == 1 || (nev1 % GGML_VEC_DOT_UNROLL != 0)) {
      for (int64_t ic = 0; ic < nev1; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ggml_vec_dot_f16(nek1, (float *)((char *)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                         (ggml_fp16_t *)((char *)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S16);
      }
    } else {
      for (int64_t ic = 0; ic < nev1; ic += GGML_VEC_DOT_UNROLL) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ggml_vec_dot_f16_unroll(nek1, nbv1, (float *)((char *)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                                ((char *)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S16);
      }
    }
  }
}

static void ggml_compute_forward_flash_attn(const struct ggml_compute_params *params, const struct ggml_tensor *q,
                                            const struct ggml_tensor *k, const struct ggml_tensor *v, const bool masked,
                                            struct ggml_tensor *dst) {
  switch (q->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_flash_attn_f16(params, q, k, v, masked, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_flash_attn_f32(params, q, k, v, masked, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_flash_ff

static void ggml_compute_forward_flash_ff_f16(const struct ggml_compute_params *params,
                                              const struct ggml_tensor *a,   // F16
                                              const struct ggml_tensor *b0,  // F16 fc_w
                                              const struct ggml_tensor *b1,  // F32 fc_b
                                              const struct ggml_tensor *c0,  // F16 proj_w
                                              const struct ggml_tensor *c1,  // F32 proj_b
                                              struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, nea, a, ne);
  GGML_TENSOR_LOCALS(size_t, nba, a, nb);
  GGML_TENSOR_LOCALS(int64_t, neb0, b0, ne);
  GGML_TENSOR_LOCALS(size_t, nbb0, b0, nb);
  GGML_TENSOR_LOCALS(int64_t, neb1, b1, ne);
  GGML_TENSOR_LOCALS(size_t, nbb1, b1, nb);
  GGML_TENSOR_LOCALS(int64_t, nec0, c0, ne);
  GGML_TENSOR_LOCALS(size_t, nbc0, c0, nb);
  GGML_TENSOR_LOCALS(int64_t, nec1, c1, ne);
  GGML_TENSOR_LOCALS(size_t, nbc1, c1, nb);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t M = neb01;

  // dst cannot be transposed or permuted

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by a rows using ggml_vec_dot_f32

  // total rows in a
  const int nr = nea1 * nea2 * nea3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // a indices
    const int ia3 = ir / (nea2 * nea1);
    const int ia2 = (ir - ia3 * nea2 * nea1) / nea1;
    const int ia1 = (ir - ia3 * nea2 * nea1 - ia2 * nea1);

    float *S = (float *)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32);

    for (int64_t ic = 0; ic < neb01; ++ic) {
      // b0 indices
      const int ib03 = ia3;
      const int ib02 = ia2;
      const int ib01 = ic;

      // S indices
      const int i1 = ib01;

      ggml_vec_dot_f16(nea0, S + i1, (ggml_fp16_t *)((char *)b0->data + (ib01 * nbb01 + ib02 * nbb02 + ib03 * nbb03)),
                       (ggml_fp16_t *)((char *)a->data + (ia1 * nba1 + ia2 * nba2 + ia3 * nba3)));
    }

    ggml_vec_add_f32(neb01, S, S, (float *)b1->data);
    // ggml_vec_gelu_f32(neb01, S, S);

    ggml_fp16_t *S16 = (ggml_fp16_t *)((float *)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32) + M);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = S[i];
    }

    ggml_vec_gelu_f16(neb01, S16, S16);

    {
      // dst indices
      const int i1 = ia1;
      const int i2 = ia2;
      const int i3 = ia3;

      for (int64_t ic = 0; ic < nec01; ++ic) {
        ggml_vec_dot_f16(neb01, (float *)((char *)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                         (ggml_fp16_t *)((char *)c0->data + (ic * nbc01 + i2 * nbc02 + i3 * nbc03)), S16);
      }

      ggml_vec_add_f32(nec01, (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)), (float *)c1->data);
    }
  }
}

static void ggml_compute_forward_flash_ff(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                          const struct ggml_tensor *b0, const struct ggml_tensor *b1,
                                          const struct ggml_tensor *c0, const struct ggml_tensor *c1,
                                          struct ggml_tensor *dst) {
  switch (b0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
    } break;
    case GGML_TYPE_F32: {
      abort();  // TODO
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_flash_attn_back

static void ggml_compute_forward_flash_attn_back_f32(const struct ggml_compute_params *params,
                                                     const struct ggml_tensor *q, const struct ggml_tensor *k,
                                                     const struct ggml_tensor *v, const struct ggml_tensor *d,
                                                     const bool masked, struct ggml_tensor *dst) {
  int64_t t0 = ggml_perf_time_us();
  UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb);
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb);
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne);
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb);
  GGML_TENSOR_LOCALS(int64_t, ned, d, ne);
  GGML_TENSOR_LOCALS(size_t, nbd, d, nb);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);
  const int mxDM = MAX(D, Mup);

  // dst cannot be transposed or permuted

  if (params->type == GGML_TASK_INIT) {
    if (ith == 0) {
      memset(dst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
    }
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2);
    const int iq2 = ir - iq3 * neq2;
    for (int iq1 = 0; iq1 < neq1; ++iq1) {
      // not sure about CACHE_LINE_SIZE_F32..
      // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
      float *S = (float *)params->wdata + ith * 2 * (mxDM + CACHE_LINE_SIZE_F32) + 0 * (mxDM + CACHE_LINE_SIZE_F32);
      float *SM = (float *)params->wdata + ith * 2 * (mxDM + CACHE_LINE_SIZE_F32) + 1 * (mxDM + CACHE_LINE_SIZE_F32);

      for (int i = M; i < Mup; ++i) {
        S[i] = -INFINITY;
      }

      for (int64_t ic = 0; ic < nek1; ++ic) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ggml_vec_dot_f32(neq0, S + i1, (float *)((char *)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                         (float *)((char *)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }

      // scale
      ggml_vec_scale_f32(nek1, S, scale);

      if (masked) {
        for (int64_t i = P; i < M; i++) {
          if (i > P + iq1) {
            S[i] = -INFINITY;
          }
        }
      }

      // softmax
      {
        float max = -INFINITY;
        ggml_vec_max_f32(M, &max, S);

        ggml_float sum = 0.0;
        {
#ifdef GGML_SOFT_MAX_ACCELERATE
          max = -max;
          vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
          vvexpf(SM, SM, &Mup);
          ggml_vec_sum_f32(Mup, &sum, SM);
#else
          uint16_t scvt[GGML_SOFT_MAX_UNROLL];
          ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

          for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
            float *SR = S + i;
            float *SW = SM + i;

            for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
              if (SR[j] == -INFINITY) {
                SW[j] = 0.0f;
              } else {
                ggml_fp16_t s = SR[j] - max;
                memcpy(&scvt[j], &s, sizeof(uint16_t));
                const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
                sump[j] += (ggml_float)val;
                SW[j] = val;
              }
            }
          }

          for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
            sum += sump[i];
          }
#endif
        }

        sum = 1.0 / sum;
        ggml_vec_scale_f32(M, SM, sum);
      }

      // step-by-step explanation
      {
        // forward-process                   shape      grads from backward process
        // parallel_for iq2,iq3:
        //  k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,iq2,iq3]  += grad[kcur]
        //  q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
        //  v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iq2,iq3]  += grad[vcur]
        //  for iq1:
        //   kcur   = k[:D,:M,iq2,iq3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
        //   qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
        //   vcur   = v[:M,:D,iq2,iq3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
        //   S0     = -Inf                   [D,1,1,1]
        //  ~S1[i]  = dot(kcur[:D,i], qcur)
        //   S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
        //   S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
        //   S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
        //   S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
        //  ~S5[i]  = dot(vcur[:,i], S4)
        //   S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,iq1,iq2,iq3]
        //  ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
        //   dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,iq1,iq2,iq3]
        // dst                               backward-/ grad[dst]                 = d
        //
        // output gradients with their dependencies:
        //
        // grad[kcur] = grad[S1].T @ qcur
        // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
        // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
        // grad[S4]   = grad[S5] @ vcur
        // grad[S4]   = d[:D,iq1,iq2,iq3] @ vcur
        // grad[qcur] = grad[S1]   @ kcur
        // grad[vcur] = grad[S5].T @ S4
        // grad[vcur] = d[:D,iq1,iq2,iq3].T @ S4
        //
        // in post-order:
        //
        // S1         = qcur @ kcur.T
        // S2         = S1 * scale
        // S3         = diag_mask_inf(S2, P)
        // S4         = softmax(S3)
        // grad[S4]   = d[:D,iq1,iq2,iq3] @ vcur
        // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
        // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
        // grad[qcur] = grad[S1]   @ kcur
        // grad[kcur] = grad[S1].T @ qcur
        // grad[vcur] = d[:D,iq1,iq2,iq3].T @ S4
        //
        // using less variables (SM=S4):
        //
        // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
        // SM            = softmax(S)
        // S             = d[:D,iq1,iq2,iq3] @ vcur
        // dot_SM_gradSM = dot(SM, S)
        // S             = SM * (S - dot(SM, S))
        // S             = diag_mask_zero(S, P) * scale
        //
        // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
        // grad[k][:D,:M,iq2,iq3]  += S.T @ qcur
        // grad[v][:M,:D,iq2,iq3]  += d[:D,iq1,iq2,iq3].T @ SM
      }

      // S = gradSM = d[:D,iq1,iq2,iq3] @ vcur
      // S = d[:D,iq1,iq2,iq3] @ vcur
      // S[:M] += vcur[:M,ic] * d[ic,iq1,iq2,iq3]
      ggml_vec_set_f32(M, S, 0);
      for (int64_t ic = 0; ic < D; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ggml_vec_mad_f32(M, S, (float *)((char *)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)),
                         *(float *)((char *)d->data + (ic * nbd0 + i1 * nbd1 + i2 * nbd2 + i3 * nbd3)));
      }

      // S = SM * (S - dot(SM, S))
      float dot_SM_gradSM = 0;
      ggml_vec_dot_f32(M, &dot_SM_gradSM, SM, S);
      ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
      ggml_vec_mul_f32(M, S, S, SM);

      // S = diag_mask_zero(S, P) * scale
      if (masked) {
        for (int64_t i = P; i < M; i++) {
          if (i > P + iq1) {
            S[i] = 0;
          }
        }
      }
      ggml_vec_scale_f32(M, S, scale);

      void *grad_q = (char *)dst->data;
      void *grad_k = (char *)dst->data + nb0 * D * N * neq2 * neq3;
      void *grad_v = (char *)dst->data + nb0 * D * N * neq2 * neq3 + nb0 * D * M * neq2 * neq3;

      const size_t nbgq1 = nb0 * neq0;
      const size_t nbgq2 = nb0 * neq0 * neq1;
      const size_t nbgq3 = nb0 * neq0 * neq1 * neq2;

      const size_t nbgk1 = nb0 * nek0;
      const size_t nbgk2 = nb0 * nek0 * nek1;
      const size_t nbgk3 = nb0 * nek0 * nek1 * neq2;

      const size_t nbgv1 = nb0 * nev0;
      const size_t nbgv2 = nb0 * nev0 * nev1;
      const size_t nbgv3 = nb0 * nev0 * nev1 * neq2;

      // S    shape [M,1]
      // SM   shape [M,1]
      // kcur shape [D,M]
      // qcur shape [D,1]
      // vcur shape [M,D]
      //
      // grad[q][:D,iq1,iq2,iq3] += S @ kcur
      // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
      // grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic]
      //
      //// grad[q][ic,iq1,iq2,iq3] += dot(kcur[:,ic],S.T)
      //// grad[q][ic,iq1,iq2,iq3] += dot(k[:D,ic,iq2,iq3],S.T)
      for (int64_t ic = 0; ic < M; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ggml_vec_mad_f32(D, (float *)((char *)grad_q + (i1 * nbgq1 + i2 * nbgq2 + i3 * nbgq3)),
                         (float *)((char *)k->data + (ic * nbk1 + i2 * nbk2 + i3 * nbk3)), S[ic]);
      }

      // grad[k][:D,:M,iq2,iq3] += S.T       @ qcur
      // grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
      // grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
      for (int64_t ic = 0; ic < M; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // ggml_vec_set_f32(D,
        //         (float *) ((char *) grad_k  + (ic*nbgk1  + i2*nbgk2  + i3*nbgk3)),
        //         0);
        ggml_vec_mad_f32(D, (float *)((char *)grad_k + (ic * nbgk1 + i2 * nbgk2 + i3 * nbgk3)),
                         (float *)((char *)q->data + (i1 * nbq1 + i2 * nbq2 + i3 * nbq3)), S[ic]);
      }

      for (int64_t ic = 0; ic < D; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ggml_vec_mad_f32(M, (float *)((char *)grad_v + (ic * nbgv1 + i2 * nbgv2 + i3 * nbgv3)), SM,
                         *(float *)((char *)d->data + (ic * nbd0 + i1 * nbd1 + i2 * nbd2 + i3 * nbd3)));
      }
    }
  }
}

static void ggml_compute_forward_flash_attn_back(const struct ggml_compute_params *params, const struct ggml_tensor *q,
                                                 const struct ggml_tensor *k, const struct ggml_tensor *v,
                                                 const struct ggml_tensor *d, const bool masked,
                                                 struct ggml_tensor *dst) {
  switch (q->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_flash_attn_back_f32(params, q, k, v, d, masked, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_win_part

static void ggml_compute_forward_win_part_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                              const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);

  const int32_t nep0 = ((const int32_t *)(opt0->data))[0];
  const int32_t nep1 = ((const int32_t *)(opt0->data))[1];
  const int32_t w = ((const int32_t *)(opt0->data))[2];

  // TODO: optimize / multi-thread
  for (int py = 0; py < nep1; ++py) {
    for (int px = 0; px < nep0; ++px) {
      const int64_t i3 = py * nep0 + px;
      for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
          for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const int64_t i02 = py * w + i2;
            const int64_t i01 = px * w + i1;
            const int64_t i00 = i0;

            const int64_t i = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0;
            const int64_t j = i02 * ne01 * ne00 + i01 * ne00 + i00;

            if (py * w + i2 >= ne02 || px * w + i1 >= ne01) {
              ((float *)dst->data)[i] = 0.0f;
            } else {
              ((float *)dst->data)[i] = ((float *)src0->data)[j];
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_win_part(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                          const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_win_part_f32(params, src0, opt0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_win_unpart

static void ggml_compute_forward_win_unpart_f32(const struct ggml_compute_params *params,
                                                const struct ggml_tensor *src0, const struct ggml_tensor *opt0,
                                                struct ggml_tensor *dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);

  const int32_t w = ((const int32_t *)(opt0->data))[0];

  // padding
  const int px = (w - ne1 % w) % w;

  const int npx = (px + ne1) / w;

  // TODO: optimize / multi-thread
  for (int64_t i2 = 0; i2 < ne2; ++i2) {
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
      for (int64_t i0 = 0; i0 < ne0; ++i0) {
        const int ip2 = i2 / w;
        const int ip1 = i1 / w;

        const int64_t i02 = i2 % w;
        const int64_t i01 = i1 % w;
        const int64_t i00 = i0;

        const int64_t i = (ip2 * npx + ip1) * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00;
        const int64_t j = i2 * ne1 * ne0 + i1 * ne0 + i0;

        ((float *)dst->data)[j] = ((float *)src0->data)[i];
      }
    }
  }
}

static void ggml_compute_forward_win_unpart(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_win_unpart_f32(params, src0, opt0, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_map_unary

static void ggml_compute_forward_map_unary_f32(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                               struct ggml_tensor *dst, const ggml_unary_op_f32_t fun) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    fun(nc, (float *)((char *)dst->data + i * (dst->nb[1])), (float *)((char *)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_map_unary(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                           struct ggml_tensor *dst, const ggml_unary_op_f32_t fun) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_unary_f32(params, src0, dst, fun);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_map_binary

static void ggml_compute_forward_map_binary_f32(const struct ggml_compute_params *params,
                                                const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                struct ggml_tensor *dst, const ggml_binary_op_f32_t fun) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = 0; i < n; i++) {
    fun(nc, (float *)((char *)dst->data + i * (dst->nb[1])), (float *)((char *)src0->data + i * (src0->nb[1])),
        (float *)((char *)src1->data + i * (src1->nb[1])));
  }
}

static void ggml_compute_forward_map_binary(const struct ggml_compute_params *params, const struct ggml_tensor *src0,
                                            const struct ggml_tensor *src1, struct ggml_tensor *dst,
                                            const ggml_binary_op_f32_t fun) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_binary_f32(params, src0, src1, dst, fun);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_map_custom1

static void ggml_compute_forward_map_custom1_f32(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                                 struct ggml_tensor *dst, const ggml_custom1_op_f32_t fun) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a);
}

static void ggml_compute_forward_map_custom1(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                             struct ggml_tensor *dst, const ggml_custom1_op_f32_t fun) {
  switch (a->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_custom1_f32(params, a, dst, fun);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_map_custom2

static void ggml_compute_forward_map_custom2_f32(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                                 const struct ggml_tensor *b, struct ggml_tensor *dst,
                                                 const ggml_custom2_op_f32_t fun) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a, b);
}

static void ggml_compute_forward_map_custom2(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                             const struct ggml_tensor *b, struct ggml_tensor *dst,
                                             const ggml_custom2_op_f32_t fun) {
  switch (a->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_custom2_f32(params, a, b, dst, fun);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_map_custom3

static void ggml_compute_forward_map_custom3_f32(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                                 const struct ggml_tensor *b, const struct ggml_tensor *c,
                                                 struct ggml_tensor *dst, const ggml_custom3_op_f32_t fun) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a, b, c);
}

static void ggml_compute_forward_map_custom3(const struct ggml_compute_params *params, const struct ggml_tensor *a,
                                             const struct ggml_tensor *b, const struct ggml_tensor *c,
                                             struct ggml_tensor *dst, const ggml_custom3_op_f32_t fun) {
  switch (a->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_custom3_f32(params, a, b, c, dst, fun);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_cross_entropy_loss

static void ggml_compute_forward_cross_entropy_loss_f32(const struct ggml_compute_params *params,
                                                        const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                        struct ggml_tensor *dst) {
  const int ith = params->ith;
  const int nth = params->nth;

  float *sums = (float *)params->wdata;

  // TODO: handle transposed/permuted matrices
  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  if (params->type == GGML_TASK_INIT) {
    if (ith == 0) {
      memset(sums, 0, sizeof(float) * (nth + nth * nc));
    }
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    if (ith == 0) {
      float *dp = (float *)dst->data;
      ggml_vec_sum_f32(nth, dp, sums);
      dp[0] *= -1.0f;
    }
    return;
  }

  const double eps = 1e-9;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float *s0 = (float *)((char *)src0->data + i1 * src0->nb[1]);
    float *s1 = (float *)((char *)src1->data + i1 * src1->nb[1]);
    float *st = (float *)params->wdata + nth + ith * nc;

    // soft_max
    ggml_float sum = 0.0;
    {
      float max = -INFINITY;
      ggml_vec_max_f32(nc, &max, s0);

      uint16_t scvt;
      for (int i = 0; i < nc; i++) {
        if (s0[i] == -INFINITY) {
          st[i] = 0.0f;
        } else {
          ggml_fp16_t s = s0[i] - max;
          memcpy(&scvt, &s, sizeof(scvt));
          const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
          sum += (ggml_float)val;
          st[i] = val;
        }
      }
    }
    // avoid log(0) by rescaling from [0..1] to [eps..1]
    sum = (1.0 - eps) / sum;
    ggml_vec_scale_f32(nc, st, sum);
    ggml_vec_add1_f32(nc, st, st, eps);
    ggml_vec_log_f32(nc, st, st);
    ggml_vec_mul_f32(nc, st, st, s1);

    ggml_vec_sum_f32(nc, sums + ith, st);
  }
}

static void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params *params,
                                                    const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                    struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_cross_entropy_loss_f32(params, src0, src1, dst);
    } break;
    default: {
    } break;
  }
}

// ggml_compute_forward_cross_entropy_loss_back

static void ggml_compute_forward_cross_entropy_loss_back_f32(const struct ggml_compute_params *params,
                                                             const struct ggml_tensor *src0,
                                                             const struct ggml_tensor *src1,
                                                             const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  const int64_t ith = params->ith;
  const int64_t nth = params->nth;

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const float eps = 1e-9f;

  // TODO: handle transposed/permuted matrices
  const int64_t nc = src0->ne[0];
  const int64_t nr = ggml_nrows(src0);

  // rows per thread
  const int64_t dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = MIN(ir0 + dr, nr);

  float *d = (float *)opt0->data;

  for (int64_t i1 = ir0; i1 < ir1; i1++) {
    float *ds0 = (float *)((char *)dst->data + i1 * dst->nb[1]);
    float *s0 = (float *)((char *)src0->data + i1 * src0->nb[1]);
    float *s1 = (float *)((char *)src1->data + i1 * src1->nb[1]);
    float *sm = (float *)params->wdata + ith * nc;

    // step by step explanation:
    {
      // float * sums = (float *) params->wdata;

      // forward pass with annotated gradients from backward pass
      // (built by going in reverse operation order, adding to gradients of current operation args)
      // st0 = exp(s0-max(s0))                                                       grad[st0] = grad[st1]*(1.0 -
      // eps)/sum from softmax_back:            grad[s0]  = st1_k * (grad[st1]_k - dot(st1, grad[st1]))
      // ggml_vec_scale_f32(nc, st, sum);           // st1 = st0*/sum = softmax(s0)  grad[st1] = grad[st2]*(1.0 - eps)
      // ggml_vec_scale_f32(nc, st, (1.0f - eps));  // st2 = st1*(1.0 - eps)         grad[st2] = grad[st3]
      // ggml_vec_add1_f32(nc, st, st, eps);        // st3 = st2 + eps               grad[st3] = grad[st4]/st3
      // ggml_vec_log_f32(nc, st, st);              // st4 = log(st3)                grad[st4] = grad[st5] * s1
      // ggml_vec_mul_f32(nc, st, st, s1);          // st5 = st4 * s1                grad[st5] = grad[sums[ith]]
      // ggml_vec_sum_f32(nc, sums + ith, st);      // sums[ith] = st5               grad[sums[ith]] =
      // grad[cross_entropy_loss] = -grad[cel]

      // substitute into grad[st1], because we can reuse softmax_back from this point on
      // grad[st1] = -grad[cel]*s1*(1.0 - eps)/(eps + softmax(s0)*(1.0 - eps))
      // postorder:
      // grad[st1] := softmax(s0)
      // grad[st1] := grad[st1]*(1.0 - eps)
      // grad[st1] := grad[st1] + eps
      // grad[st1] := s1 / grad[st1]
      // grad[st1] := grad[st1]*(1.0-eps)*-grad[cel]

      // src0 gradients by going through softmax_back
      // grad[s0] = st1_k * (grad[st1]_k - dot(st1, grad[st1]))
      //   from softmax_back:
      //   dxk = yk * (dyk - dot(y, dy))
      //   dot_y_dy := dot(y, dy)
      //   dx := dy
      //   dx := dx - dot_y_dy
      //   dx := dx * y
      //   postorder:
      //   dot_st1_dst1 := dot(st1, grad[st1])
      //   grad[s0] := grad[st1]
      //   grad[s0] := grad[s0] - dot_st1_dst1
      //   grad[s0] := grad[s0] * st1

      // prepend postorder from grad[st1] directly using grad[s0] as memory location, as we will grad[s0] := grad[st1]
      // sm           := softmax(s0)
      // grad[s0]     := sm*(1.0 - eps)
      // grad[s0]     := grad[s0] + eps
      // grad[s0]     := s1 / grad[s0]
      // grad[s0]     := grad[s0]*(1.0-eps)*-grad[cel]
      // dot_st1_dst1 := dot(sm, grad[s0])
      // grad[s0]     := grad[s0] - dot_st1_dst1
      // grad[s0]     := grad[s0] * sm
    }

    // soft_max
    ggml_float sum = 0.0;
    {
      float max = -INFINITY;
      ggml_vec_max_f32(nc, &max, s0);

      uint16_t scvt;
      for (int i = 0; i < nc; i++) {
        if (s0[i] == -INFINITY) {
          sm[i] = 0.0f;
        } else {
          // const float val = (s0[i] == -INFINITY) ? 0.0 : exp(s0[i] - max);
          ggml_fp16_t s = s0[i] - max;
          memcpy(&scvt, &s, sizeof(scvt));
          const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
          sum += (ggml_float)val;
          sm[i] = val;
        }
      }

      sum = 1.0 / sum;
    }

    float dot_st1_dst1 = 0;
    ggml_vec_scale_f32(nc, sm, sum);
    ggml_vec_cpy_f32(nc, ds0, sm);
    ggml_vec_scale_f32(nc, ds0, (1.0f - eps));
    ggml_vec_add1_f32(nc, ds0, ds0, eps);
    ggml_vec_div_f32(nc, ds0, s1, ds0);
    ggml_vec_scale_f32(nc, ds0, -(1.0f - eps) * d[0]);
    ggml_vec_dot_f32(nc, &dot_st1_dst1, sm, ds0);
    ggml_vec_acc1_f32(nc, ds0, -dot_st1_dst1);
    ggml_vec_mul_f32(nc, ds0, ds0, sm);
  }
}

static void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params *params,
                                                         const struct ggml_tensor *src0, const struct ggml_tensor *src1,
                                                         const struct ggml_tensor *opt0, struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_cross_entropy_loss_back_f32(params, src0, src1, opt0, dst);
    } break;
    default: {
    } break;
  }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params *params, struct ggml_tensor *tensor) {
  switch (tensor->op) {
    case GGML_OP_DUP: {
      ggml_compute_forward_dup(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ADD: {
      ggml_compute_forward_add(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ADD1: {
      ggml_compute_forward_add1(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ACC: {
      ggml_compute_forward_acc(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_SUB: {
      ggml_compute_forward_sub(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_MUL: {
      ggml_compute_forward_mul(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_DIV: {
      ggml_compute_forward_div(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SQR: {
      ggml_compute_forward_sqr(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SQRT: {
      ggml_compute_forward_sqrt(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_LOG: {
      ggml_compute_forward_log(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SUM: {
      ggml_compute_forward_sum(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SUM_ROWS: {
      ggml_compute_forward_sum_rows(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_MEAN: {
      ggml_compute_forward_mean(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ARGMAX: {
      ggml_compute_forward_argmax(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_REPEAT: {
      ggml_compute_forward_repeat(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_REPEAT_BACK: {
      ggml_compute_forward_repeat_back(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ABS: {
      ggml_compute_forward_abs(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SGN: {
      ggml_compute_forward_sgn(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_NEG: {
      ggml_compute_forward_neg(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_STEP: {
      ggml_compute_forward_step(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_TANH: {
      ggml_compute_forward_tanh(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ELU: {
      ggml_compute_forward_elu(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RELU: {
      ggml_compute_forward_relu(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_GELU: {
      ggml_compute_forward_gelu(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_GELU_QUICK: {
      ggml_compute_forward_gelu_quick(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SILU: {
      ggml_compute_forward_silu(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SILU_BACK: {
      ggml_compute_forward_silu_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_NORM: {
      ggml_compute_forward_norm(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RMS_NORM: {
      ggml_compute_forward_rms_norm(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RMS_NORM_BACK: {
      ggml_compute_forward_rms_norm_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_MUL_MAT: {
      ggml_compute_forward_mul_mat(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_OUT_PROD: {
      ggml_compute_forward_out_prod(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SCALE: {
      ggml_compute_forward_scale(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SET: {
      ggml_compute_forward_set(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_CPY: {
      ggml_compute_forward_cpy(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_CONT: {
      ggml_compute_forward_cont(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RESHAPE: {
      ggml_compute_forward_reshape(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_VIEW: {
      ggml_compute_forward_view(params, tensor->src[0]);
    } break;
    case GGML_OP_PERMUTE: {
      ggml_compute_forward_permute(params, tensor->src[0]);
    } break;
    case GGML_OP_TRANSPOSE: {
      ggml_compute_forward_transpose(params, tensor->src[0]);
    } break;
    case GGML_OP_GET_ROWS: {
      ggml_compute_forward_get_rows(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_GET_ROWS_BACK: {
      ggml_compute_forward_get_rows_back(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_DIAG: {
      ggml_compute_forward_diag(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_DIAG_MASK_INF: {
      ggml_compute_forward_diag_mask_inf(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_DIAG_MASK_ZERO: {
      ggml_compute_forward_diag_mask_zero(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SOFT_MAX: {
      ggml_compute_forward_soft_max(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SOFT_MAX_BACK: {
      ggml_compute_forward_soft_max_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ROPE: {
      ggml_compute_forward_rope(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ROPE_BACK: {
      ggml_compute_forward_rope_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ALIBI: {
      ggml_compute_forward_alibi(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CLAMP: {
      ggml_compute_forward_clamp(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CONV_1D: {
      ggml_compute_forward_conv_1d(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_CONV_2D: {
      ggml_compute_forward_conv_2d(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_POOL_1D: {
      ggml_compute_forward_pool_1d(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_POOL_2D: {
      ggml_compute_forward_pool_2d(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_FLASH_ATTN: {
      const int32_t t = ggml_get_i32_1d(tensor->src[3], 0);

      const bool masked = t != 0;
      ggml_compute_forward_flash_attn(params, tensor->src[0], tensor->src[1], tensor->src[2], masked, tensor);
    } break;
    case GGML_OP_FLASH_FF: {
      ggml_compute_forward_flash_ff(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3],
                                    tensor->src[4], tensor);
    } break;
    case GGML_OP_FLASH_ATTN_BACK: {
      int32_t t = ggml_get_i32_1d(tensor->src[4], 0);

      bool masked = t != 0;
      ggml_compute_forward_flash_attn_back(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3],
                                           masked, tensor);
    } break;
    case GGML_OP_WIN_PART: {
      ggml_compute_forward_win_part(params, tensor->src[0], tensor->src[2], tensor);
    } break;
    case GGML_OP_WIN_UNPART: {
      ggml_compute_forward_win_unpart(params, tensor->src[0], tensor->src[2], tensor);
    } break;
    case GGML_OP_MAP_UNARY: {
      const ggml_unary_op_f32_t fun = *((ggml_unary_op_f32_t *)tensor->src[2]->data);
      ggml_compute_forward_map_unary(params, tensor->src[0], tensor, fun);
    } break;
    case GGML_OP_MAP_BINARY: {
      const ggml_binary_op_f32_t fun = *((ggml_binary_op_f32_t *)tensor->src[2]->data);
      ggml_compute_forward_map_binary(params, tensor->src[0], tensor->src[1], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM1: {
      const ggml_custom1_op_f32_t fun = *((ggml_custom1_op_f32_t *)tensor->src[2]->data);
      ggml_compute_forward_map_custom1(params, tensor->src[0], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM2: {
      const ggml_custom2_op_f32_t fun = *((ggml_custom2_op_f32_t *)tensor->src[2]->data);
      ggml_compute_forward_map_custom2(params, tensor->src[0], tensor->src[1], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM3: {
      const ggml_custom3_op_f32_t fun = *((ggml_custom3_op_f32_t *)tensor->src[2]->data);
      ggml_compute_forward_map_custom3(params, tensor->src[0], tensor->src[1], tensor->src[3], tensor, fun);
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS: {
      ggml_compute_forward_cross_entropy_loss(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
      ggml_compute_forward_cross_entropy_loss_back(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_NONE:
    case GGML_OP_COUNT: {
    } break;
  }
}

////////////////////////////////////////////////////////////////////////////////

static void ggml_compute_backward(struct ggml_context *ctx, struct ggml_tensor *tensor, bool inplace) {
  struct ggml_tensor *src0 = tensor->src[0];
  struct ggml_tensor *src1 = tensor->src[1];

  switch (tensor->op) {
    case GGML_OP_DUP: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_ADD: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad = ggml_add_impl(ctx, src1->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_ADD1: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad = ggml_add_impl(ctx, src1->grad,
                                   ggml_mean(ctx, tensor->grad),  // TODO: should probably be sum instead of mean
                                   inplace);
      }
    } break;
    case GGML_OP_ACC: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        const size_t nb1 = ((int32_t *)tensor->src[2]->data)[0];
        const size_t nb2 = ((int32_t *)tensor->src[2]->data)[1];
        const size_t nb3 = ((int32_t *)tensor->src[2]->data)[2];
        const size_t offset = ((int32_t *)tensor->src[2]->data)[3];

        struct ggml_tensor *tensor_grad_view =
            ggml_view_4d(ctx, tensor->grad, src1->grad->ne[0], src1->grad->ne[1], src1->grad->ne[2], src1->grad->ne[3],
                         nb1, nb2, nb3, offset);

        src1->grad =
            ggml_add_impl(ctx, src1->grad, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1->grad), inplace);
      }
    } break;
    case GGML_OP_SUB: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad = ggml_sub_impl(ctx, src1->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_MUL: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_mul(ctx, src1, tensor->grad), inplace);
      }
      if (src1->grad) {
        src1->grad = ggml_add_impl(ctx, src1->grad, ggml_mul(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case GGML_OP_DIV: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_div(ctx, tensor->grad, src1), inplace);
      }
      if (src1->grad) {
        src1->grad = ggml_sub_impl(ctx, src1->grad, ggml_mul(ctx, tensor->grad, ggml_div(ctx, tensor, src1)), inplace);
      }
    } break;
    case GGML_OP_SQR: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(
            ctx, src0->grad, ggml_scale(ctx, ggml_mul(ctx, src0, tensor->grad), ggml_new_f32(ctx, 2.0f)), inplace);
      }
    } break;
    case GGML_OP_SQRT: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(
            ctx, src0->grad, ggml_scale(ctx, ggml_div(ctx, tensor->grad, tensor), ggml_new_f32(ctx, 0.5f)), inplace);
      }
    } break;
    case GGML_OP_LOG: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_div(ctx, tensor->grad, src0), inplace);
      }
    } break;
    case GGML_OP_SUM: {
      if (src0->grad) {
        src0->grad = ggml_add1_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_SUM_ROWS: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_repeat(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX: {
      abort();  // TODO: implement
    } break;
    case GGML_OP_REPEAT: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_repeat_back(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case GGML_OP_REPEAT_BACK: {
      if (src0->grad) {
        // TODO: test this
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_repeat(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case GGML_OP_ABS: {
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_mul(ctx, ggml_sgn(ctx, src0), tensor->grad), inplace);
      }
    } break;
    case GGML_OP_SGN: {
    } break;
    case GGML_OP_NEG: {
      if (src0->grad) {
        src0->grad = ggml_sub_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_STEP: {
    } break;
    case GGML_OP_TANH:
    case GGML_OP_ELU: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_RELU: {
      if (src0->grad) {
        src0->grad = ggml_sub_impl(ctx, src0->grad, ggml_mul(ctx, ggml_step(ctx, src0), tensor->grad), inplace);
      }
    } break;
    case GGML_OP_GELU:
    case GGML_OP_GELU_QUICK: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_SILU: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_silu_back(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case GGML_OP_SILU_BACK:
    case GGML_OP_NORM: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_RMS_NORM: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_rms_norm_back(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case GGML_OP_RMS_NORM_BACK: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_MUL_MAT: {
      // https://cs231n.github.io/optimization-2/#staged
      // # forward pass
      // s0 = np.random.randn(5, 10)
      // s1 = np.random.randn(10, 3)
      // t = s0.dot(s1)

      // # now suppose we had the gradient on t from above in the circuit
      // dt = np.random.randn(*t.shape) # same shape as t
      // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
      // ds1 = t.T.dot(dt)

      // tensor.shape [m,p]
      // src0.shape   [n,m]
      // src1.shape   [n,p]

      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad,
                                   ggml_out_prod(ctx,            // [n,m]
                                                 src1,           // [n,p]
                                                 tensor->grad),  // [m,p]
                                   inplace);
      }
      if (src1->grad) {
        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
        // avoid transpose of src0, rather transpose smaller tensor->grad
        // and then use ggml_out_prod
        // [n,p]
        // [n,m]
        // [p,m]
        // [m,p]
        src1->grad =
            ggml_add_impl(ctx, src1->grad, ggml_out_prod(ctx, src0, ggml_transpose(ctx, tensor->grad)), inplace);
      }
    } break;
    case GGML_OP_OUT_PROD: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_SCALE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_scale_impl(ctx, tensor->grad, src1, false), inplace);
      }
      if (src1->grad) {
        src1->grad =
            ggml_add_impl(ctx, src1->grad, ggml_sum(ctx, ggml_mul_impl(ctx, tensor->grad, src0, false)), inplace);
      }
    } break;
    case GGML_OP_SET: {
      const size_t nb1 = ((int32_t *)tensor->src[2]->data)[0];
      const size_t nb2 = ((int32_t *)tensor->src[2]->data)[1];
      const size_t nb3 = ((int32_t *)tensor->src[2]->data)[2];
      const size_t offset = ((int32_t *)tensor->src[2]->data)[3];

      struct ggml_tensor *tensor_grad_view = NULL;

      if (src0->grad || src1->grad) {
        tensor_grad_view = ggml_view_4d(ctx, tensor->grad, src1->grad->ne[0], src1->grad->ne[1], src1->grad->ne[2],
                                        src1->grad->ne[3], nb1, nb2, nb3, offset);
      }

      if (src0->grad) {
        src0->grad = ggml_add_impl(
            ctx, src0->grad,
            ggml_acc_impl(ctx, tensor->grad, ggml_neg(ctx, tensor_grad_view), nb1, nb2, nb3, offset, false), inplace);
      }

      if (src1->grad) {
        src1->grad =
            ggml_add_impl(ctx, src1->grad, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1->grad), inplace);
      }
    } break;
    case GGML_OP_CPY:
    case GGML_OP_CONT: {
      // necessary for llama
      // cpy overwrites value of src1 by src0 and returns view(src1)
      // the overwriting is mathematically equivalent to:
      // tensor = src0 * 1 + src1 * 0
      if (src0->grad) {
        // dsrc0 = dtensor * 1
        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case GGML_OP_RESHAPE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_reshape(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case GGML_OP_VIEW: {
      // necessary for llama
      if (src0->grad) {
        size_t offset;

        memcpy(&offset, tensor->src[2]->data, sizeof(offset));

        size_t nb1 = tensor->nb[1];
        size_t nb2 = tensor->nb[2];
        size_t nb3 = tensor->nb[3];

        if (src0->type != src0->grad->type) {
          // gradient is typically F32, but src0 could be other type
          size_t ng = ggml_element_size(src0->grad);
          size_t n0 = ggml_element_size(src0);

          offset = (offset / n0) * ng;
          nb1 = (nb1 / n0) * ng;
          nb2 = (nb2 / n0) * ng;
          nb3 = (nb3 / n0) * ng;
        }

        src0->grad = ggml_acc_impl(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, inplace);
      }
    } break;
    case GGML_OP_PERMUTE: {
      // necessary for llama
      if (src0->grad) {
        int32_t *axes = (int32_t *)tensor->src[2]->data;
        int axis0 = axes[0] & 0x3;
        int axis1 = axes[1] & 0x3;
        int axis2 = axes[2] & 0x3;
        int axis3 = axes[3] & 0x3;
        int axes_backward[4] = {0, 0, 0, 0};
        axes_backward[axis0] = 0;
        axes_backward[axis1] = 1;
        axes_backward[axis2] = 2;
        axes_backward[axis3] = 3;
        src0->grad = ggml_add_impl(
            ctx, src0->grad,
            ggml_permute(ctx, tensor->grad, axes_backward[0], axes_backward[1], axes_backward[2], axes_backward[3]),
            inplace);
      }
    } break;
    case GGML_OP_TRANSPOSE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_transpose(ctx, tensor->grad), inplace);
      }
    } break;
    case GGML_OP_GET_ROWS: {
      // necessary for llama (only for tokenizer)
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_get_rows_back(ctx, tensor->grad, src1, src0->grad), inplace);
      }
    } break;
    case GGML_OP_GET_ROWS_BACK:
    case GGML_OP_DIAG: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_DIAG_MASK_INF: {
      // necessary for llama
      if (src0->grad) {
        const int n_past = ((int32_t *)src1->data)[0];
        src0->grad =
            ggml_add_impl(ctx, src0->grad, ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false), inplace);
      }
    } break;
    case GGML_OP_DIAG_MASK_ZERO: {
      // necessary for llama
      if (src0->grad) {
        const int n_past = ((int32_t *)src1->data)[0];
        src0->grad =
            ggml_add_impl(ctx, src0->grad, ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false), inplace);
      }
    } break;
    case GGML_OP_SOFT_MAX: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_soft_max_back(ctx, tensor->grad, tensor), inplace);
      }

    } break;
    case GGML_OP_SOFT_MAX_BACK: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_ROPE: {
      // necessary for llama
      if (src0->grad) {
        const int n_past = ((int32_t *)src1->data)[0];
        const int n_dims = ((int32_t *)src1->data)[1];
        const int mode = ((int32_t *)src1->data)[2];
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_rope_back(ctx, tensor->grad, n_past, n_dims, mode), inplace);
      }
    } break;
    case GGML_OP_ROPE_BACK: {
      if (src0->grad) {
        const int n_past = ((int32_t *)src1->data)[0];
        const int n_dims = ((int32_t *)src1->data)[1];
        const int mode = ((int32_t *)src1->data)[2];
        const int n_ctx = ((int32_t *)src1->data)[3];
        src0->grad = ggml_add_impl(ctx, src0->grad, ggml_rope(ctx, tensor->grad, n_past, n_dims, mode, n_ctx), inplace);
      }
    } break;
    case GGML_OP_ALIBI:
    case GGML_OP_CLAMP:
    case GGML_OP_CONV_1D:
    case GGML_OP_CONV_2D:
    case GGML_OP_POOL_1D:
    case GGML_OP_POOL_2D: {
      abort();  // TODO: not implemented
    } break;
    case GGML_OP_FLASH_ATTN: {
      struct ggml_tensor *flash_grad = NULL;
      if (src0->grad || src1->grad || tensor->src[2]->grad) {
        int32_t t = ggml_get_i32_1d(tensor->src[3], 0);

        bool masked = t != 0;
        flash_grad = ggml_flash_attn_back(ctx, src0, src1, tensor->src[2], tensor->grad, masked);
      }

      if (src0->grad) {
        struct ggml_tensor *grad_q = NULL;
        const size_t nb0 = flash_grad->nb[0];
        const size_t offset = 0;
        switch (src0->n_dims) {
          case 2: {
            grad_q = ggml_view_2d(ctx, flash_grad, src0->ne[0], src0->ne[1], nb0 * src0->ne[0], offset);
          } break;
          case 3: {
            grad_q = ggml_view_3d(ctx, flash_grad, src0->ne[0], src0->ne[1], src0->ne[2], nb0 * src0->ne[0],
                                  nb0 * src0->ne[0] * src0->ne[1], offset);
          } break;
          case 4: {
            grad_q =
                ggml_view_4d(ctx, flash_grad, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], nb0 * src0->ne[0],
                             nb0 * src0->ne[0] * src0->ne[1], nb0 * src0->ne[0] * src0->ne[1] * src0->ne[2], offset);
          } break;
        }

        src0->grad = ggml_add_impl(ctx, src0->grad, grad_q, inplace);
      }

      if (src1->grad) {
        struct ggml_tensor *grad_k = NULL;
        const size_t nb0 = flash_grad->nb[0];
        const size_t offset = nb0 * src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
        switch (src1->n_dims) {
          case 2: {
            grad_k = ggml_view_2d(ctx, flash_grad, src1->ne[0], src1->ne[1], nb0 * src1->ne[0], offset);
          } break;
          case 3: {
            grad_k = ggml_view_3d(ctx, flash_grad, src1->ne[0], src1->ne[1], src1->ne[2], nb0 * src1->ne[0],
                                  nb0 * src1->ne[0] * src1->ne[1], offset);
          } break;
          case 4: {
            grad_k =
                ggml_view_4d(ctx, flash_grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], nb0 * src1->ne[0],
                             nb0 * src1->ne[0] * src1->ne[1], nb0 * src1->ne[0] * src1->ne[1] * src1->ne[2], offset);
          } break;
        }

        src1->grad = ggml_add_impl(ctx, src1->grad, grad_k, inplace);
      }

      struct ggml_tensor *opt0 = tensor->src[2];

      if (opt0->grad) {
        struct ggml_tensor *grad_v = NULL;
        const size_t nb0 = flash_grad->nb[0];
        const size_t offset = nb0 * src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] +
                              nb0 * src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
        switch (opt0->n_dims) {
          case 2: {
            grad_v = ggml_view_2d(ctx, flash_grad, opt0->ne[0], opt0->ne[1], nb0 * opt0->ne[0], offset);
          } break;
          case 3: {
            grad_v = ggml_view_3d(ctx, flash_grad, opt0->ne[0], opt0->ne[1], opt0->ne[2], nb0 * opt0->ne[0],
                                  nb0 * opt0->ne[0] * opt0->ne[1], offset);
          } break;
          case 4: {
            grad_v =
                ggml_view_4d(ctx, flash_grad, opt0->ne[0], opt0->ne[1], opt0->ne[2], opt0->ne[3], nb0 * opt0->ne[0],
                             nb0 * opt0->ne[0] * opt0->ne[1], nb0 * opt0->ne[0] * opt0->ne[1] * opt0->ne[2], offset);
          } break;
        }

        opt0->grad = ggml_add_impl(ctx, opt0->grad, grad_v, inplace);
      }
    } break;
    case GGML_OP_FLASH_FF:
    case GGML_OP_FLASH_ATTN_BACK:
    case GGML_OP_WIN_PART:
    case GGML_OP_WIN_UNPART:
    case GGML_OP_MAP_UNARY:
    case GGML_OP_MAP_BINARY:
    case GGML_OP_MAP_CUSTOM1:
    case GGML_OP_MAP_CUSTOM2:
    case GGML_OP_MAP_CUSTOM3: {
      abort();  // not supported
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS: {
      if (src0->grad) {
        src0->grad =
            ggml_add_impl(ctx, src0->grad, ggml_cross_entropy_loss_back(ctx, src0, src1, tensor->grad), inplace);
      }
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
      abort();  // not supported
    } break;
    case GGML_OP_NONE:
    case GGML_OP_COUNT: {
    } break;
  }
}

static void ggml_visit_parents(struct ggml_cgraph *cgraph, struct ggml_tensor *node) {
  // check if already visited
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return;
    }
  }

  for (int i = 0; i < cgraph->n_leafs; i++) {
    if (cgraph->leafs[i] == node) {
      return;
    }
  }

  for (int i = 0; i < GGML_MAX_SRC; ++i) {
    if (node->src[i]) {
      ggml_visit_parents(cgraph, node->src[i]);
    }
  }

  if (node->op == GGML_OP_NONE && node->grad == NULL) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)

    if (strlen(node->name) == 0) {
      ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
    }

    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
  } else {
    if (strlen(node->name) == 0) {
      ggml_format_name(node, "node_%d", cgraph->n_nodes);
    }

    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->grads[cgraph->n_nodes] = node->grad;
    cgraph->n_nodes++;
  }
}

static void ggml_build_forward_impl(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor, bool expand) {
  if (!expand) {
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;
  }

  const int n0 = cgraph->n_nodes;
  UNUSED(n0);

  ggml_visit_parents(cgraph, tensor);

  const int n_new = cgraph->n_nodes - n0;
  GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

  if (n_new > 0) {
    // the last added node should always be starting point
  }
}

void ggml_build_forward_expand(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor) {
  ggml_build_forward_impl(cgraph, tensor, true);
}

struct ggml_cgraph ggml_build_forward(struct ggml_tensor *tensor) {
  struct ggml_cgraph result = {
      /*.n_nodes      =*/0,
      /*.n_leafs      =*/0,
      /*.nodes        =*/{NULL},
      /*.grads        =*/{NULL},
      /*.leafs        =*/{NULL},
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
  };

  ggml_build_forward_impl(&result, tensor, false);

  return result;
}

struct ggml_cgraph ggml_build_backward(struct ggml_context *ctx, struct ggml_cgraph *gf, bool keep) {
  struct ggml_cgraph result = *gf;

  // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
  if (keep) {
    for (int i = 0; i < gf->n_nodes; i++) {
      struct ggml_tensor *node = gf->nodes[i];

      if (node->grad) {
        node->grad = ggml_dup_tensor(ctx, node);
        gf->grads[i] = node->grad;
      }
    }
  }

  for (int i = gf->n_nodes - 1; i >= 0; i--) {
    struct ggml_tensor *node = gf->nodes[i];

    // because we detached the grad nodes from the original graph, we can afford inplace operations
    if (node->grad) {
      ggml_compute_backward(ctx, node, keep);
    }
  }

  for (int i = gf->n_nodes - 1; i >= 0; i--) {
    struct ggml_tensor *node = gf->nodes[i];

    if (node->is_param) {
      GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *)node);
      ggml_build_forward_impl(&result, node->grad, true);
    }
  }

  return result;
}

//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

typedef int ggml_lock_t;

#define ggml_lock_init(x) UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x) UNUSED(x)
#define ggml_lock_unlock(x) UNUSED(x)

#define GGML_LOCK_INITIALIZER 0

#define ggml_thread_create pthread_create
#define ggml_thread_join pthread_join

// Android's libc implementation "bionic" does not support setting affinity
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
void set_numa_thread_affinity(int thread_n, int n_threads) {
  UNUSED(thread_n);
  UNUSED(n_threads);
}
void clear_numa_thread_affinity(void) {}

struct ggml_compute_state_shared {
  const struct ggml_cgraph *cgraph;
  const struct ggml_cplan *cplan;

  int64_t perf_node_start_cycles;
  int64_t perf_node_start_time_us;

  const int n_threads;

  // synchronization primitives
  atomic_int n_active;  // num active threads
  atomic_int node_n;    // active graph node

  bool (*abort_callback)(void *data);  // abort ggml_graph_compute when true
  void *abort_callback_data;
};

struct ggml_compute_state {
  pthread_t thrd;
  int ith;
  struct ggml_compute_state_shared *shared;
};

static void ggml_graph_compute_perf_stats_node(struct ggml_tensor *node, const struct ggml_compute_state_shared *st) {
  int64_t cycles_cur = ggml_perf_cycles() - st->perf_node_start_cycles;
  int64_t time_us_cur = ggml_perf_time_us() - st->perf_node_start_time_us;

  node->perf_runs++;
  node->perf_cycles += cycles_cur;
  node->perf_time_us += time_us_cur;
}

static void *ggml_graph_compute_thread(void *data) {
  struct ggml_compute_state *state = (struct ggml_compute_state *)data;

  const struct ggml_cgraph *cgraph = state->shared->cgraph;
  const struct ggml_cplan *cplan = state->shared->cplan;

  const int *n_tasks_arr = cplan->n_tasks;
  const int n_threads = state->shared->n_threads;

  set_numa_thread_affinity(state->ith, n_threads);

  int node_n = -1;

  while (true) {
    if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
      state->shared->node_n += 1;
      return (void *)GGML_EXIT_ABORTED;
    }
    if (atomic_fetch_sub(&state->shared->n_active, 1) == 1) {
      // all other threads are finished and spinning
      // do finalize and init here so we don't have synchronize again
      struct ggml_compute_params params = {
          /*.type  =*/GGML_TASK_FINALIZE,
          /*.ith   =*/0,
          /*.nth   =*/0,
          /*.wsize =*/cplan->work_size,
          /*.wdata =*/cplan->work_data,
      };

      if (node_n != -1) {
        /* FINALIZE */
        struct ggml_tensor *node = state->shared->cgraph->nodes[node_n];
        if (GGML_OP_HAS_FINALIZE[node->op]) {
          params.nth = n_tasks_arr[node_n];
          ggml_compute_forward(&params, node);
          ggml_graph_compute_perf_stats_node(node, state->shared);
        }
      }

      // distribute new work or execute it direct if 1T
      while (++node_n < cgraph->n_nodes) {
        GGML_PRINT_DEBUG_5("%s: %d/%d\n", __func__, node_n, cgraph->n_nodes);

        struct ggml_tensor *node = cgraph->nodes[node_n];
        const int n_tasks = n_tasks_arr[node_n];

        state->shared->perf_node_start_cycles = ggml_perf_cycles();
        state->shared->perf_node_start_time_us = ggml_perf_time_us();

        params.nth = n_tasks;

        /* INIT */
        if (GGML_OP_HAS_INIT[node->op]) {
          params.type = GGML_TASK_INIT;
          ggml_compute_forward(&params, node);
        }

        if (n_tasks == 1) {
          // TODO: maybe push node_n to the atomic but if other threads see n_tasks is 1,
          // they do something more efficient than spinning (?)
          params.type = GGML_TASK_COMPUTE;
          ggml_compute_forward(&params, node);

          if (GGML_OP_HAS_FINALIZE[node->op]) {
            params.type = GGML_TASK_FINALIZE;
            ggml_compute_forward(&params, node);
            ggml_graph_compute_perf_stats_node(node, state->shared);
          }
        } else {
          break;
        }

        if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
          break;
        }
      }

      atomic_store(&state->shared->n_active, n_threads);
      atomic_store(&state->shared->node_n, node_n);
    } else {
      // wait for other threads to finish
      const int last = node_n;
      do {
        // sched_yield();
        node_n = atomic_load(&state->shared->node_n);
      } while (node_n == last);
    }

    // check if we should stop
    if (node_n >= cgraph->n_nodes) {
      break;
    }

    /* COMPUTE */
    struct ggml_tensor *node = cgraph->nodes[node_n];
    const int n_tasks = n_tasks_arr[node_n];

    struct ggml_compute_params params = {
        /*.type  =*/GGML_TASK_COMPUTE,
        /*.ith   =*/state->ith,
        /*.nth   =*/n_tasks,
        /*.wsize =*/cplan->work_size,
        /*.wdata =*/cplan->work_data,
    };

    if (state->ith < n_tasks) {
      ggml_compute_forward(&params, node);
    }
  }

  return GGML_EXIT_SUCCESS;
}

struct ggml_cplan ggml_graph_plan(struct ggml_cgraph *cgraph, int n_threads) {
  if (n_threads <= 0) {
    n_threads = GGML_DEFAULT_N_THREADS;
  }

  size_t work_size = 0;

  struct ggml_cplan cplan;
  memset(&cplan, 0, sizeof(struct ggml_cplan));

  // thread scheduling for the different operations + work buffer size estimation
  for (int i = 0; i < cgraph->n_nodes; i++) {
    int n_tasks = 1;

    struct ggml_tensor *node = cgraph->nodes[i];

    switch (node->op) {
      case GGML_OP_CPY:
      case GGML_OP_DUP: {
        n_tasks = n_threads;

        size_t cur = 0;
        if (ggml_is_quantized(node->type)) {
          cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * node->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_ADD:
      case GGML_OP_ADD1: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (ggml_is_quantized(node->src[0]->type)) {
          cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * node->src[0]->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_ACC: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (ggml_is_quantized(node->src[0]->type)) {
          cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * node->src[1]->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_SUB:
      case GGML_OP_DIV:
      case GGML_OP_SQR:
      case GGML_OP_SQRT:
      case GGML_OP_LOG:
      case GGML_OP_SUM:
      case GGML_OP_SUM_ROWS:
      case GGML_OP_MEAN:
      case GGML_OP_ARGMAX:
      case GGML_OP_REPEAT:
      case GGML_OP_REPEAT_BACK:
      case GGML_OP_ABS:
      case GGML_OP_SGN:
      case GGML_OP_NEG:
      case GGML_OP_STEP:
      case GGML_OP_TANH:
      case GGML_OP_ELU:
      case GGML_OP_RELU: {
        n_tasks = 1;
      } break;
      case GGML_OP_MUL:
      case GGML_OP_GELU:
      case GGML_OP_GELU_QUICK:
      case GGML_OP_SILU:
      case GGML_OP_SILU_BACK:
      case GGML_OP_NORM:
      case GGML_OP_RMS_NORM:
      case GGML_OP_RMS_NORM_BACK: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_MUL_MAT:
      case GGML_OP_OUT_PROD: {
        n_tasks = n_threads;

        // TODO: use different scheduling for different matrix sizes

        size_t cur = 0;
        const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

        if (ggml_compute_forward_mul_mat_use_blas(node->src[0], node->src[1], node)) {
          n_tasks = 1;  // TODO: this actually is doing nothing the threads are still spinning
          if (node->src[0]->type != GGML_TYPE_F32) {
            // here we need memory just for single 2D matrix from src0
            cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * (node->src[0]->ne[0] * node->src[0]->ne[1]);
          }
        } else if (node->src[1]->type != vec_dot_type) {
          cur = GGML_TYPE_SIZE[vec_dot_type] * ggml_nelements(node->src[1]) / GGML_BLCK_SIZE[vec_dot_type];
        } else {
          cur = 0;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_SCALE:
      case GGML_OP_SET:
      case GGML_OP_CONT:
      case GGML_OP_RESHAPE:
      case GGML_OP_VIEW:
      case GGML_OP_PERMUTE:
      case GGML_OP_TRANSPOSE:
      case GGML_OP_GET_ROWS:
      case GGML_OP_GET_ROWS_BACK:
      case GGML_OP_DIAG:
      case GGML_OP_DIAG_MASK_ZERO: {
        n_tasks = 1;
      } break;
      case GGML_OP_DIAG_MASK_INF:
      case GGML_OP_SOFT_MAX:
      case GGML_OP_SOFT_MAX_BACK:
      case GGML_OP_ROPE:
      case GGML_OP_ROPE_BACK: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_ALIBI:
      case GGML_OP_CLAMP: {
        n_tasks = 1;  // TODO
      } break;
      case GGML_OP_CONV_1D: {
        n_tasks = n_threads;

        size_t cur = 0;
        const int nk = node->src[0]->ne[0];

        if (node->src[0]->type == GGML_TYPE_F16 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(ggml_fp16_t) * (nk * ggml_up32(node->src[0]->ne[1]) * node->src[0]->ne[2] +
                                       (2 * (nk / 2) + node->src[1]->ne[0]) * node->src[1]->ne[1]);
        } else if (node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * (nk * ggml_up32(node->src[0]->ne[1]) * node->src[0]->ne[2] +
                                 (2 * (nk / 2) + node->src[1]->ne[0]) * node->src[1]->ne[1]);
        } else {
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_CONV_2D: {
        n_tasks = n_threads;

        const int64_t ne00 = node->src[0]->ne[0];  // W
        const int64_t ne01 = node->src[0]->ne[1];  // H
        const int64_t ne02 = node->src[0]->ne[2];  // C
        const int64_t ne03 = node->src[0]->ne[3];  // N

        const int64_t ne10 = node->src[1]->ne[0];  // W
        const int64_t ne11 = node->src[1]->ne[1];  // H
        const int64_t ne12 = node->src[1]->ne[2];  // C

        const int64_t nk = ne00 * ne01;

        UNUSED(ne02);
        UNUSED(ne03);
        UNUSED(nk);

        size_t cur = 0;

        if (node->src[0]->type == GGML_TYPE_F16 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(ggml_fp16_t) * (ne10 * ne11 * ne12);
        } else if (node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * (ne10 * ne11 * ne12);
        } else {
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_POOL_1D:
      case GGML_OP_POOL_2D: {
        n_tasks = 1;
      } break;
      case GGML_OP_FLASH_ATTN: {
        n_tasks = n_threads;

        size_t cur = 0;

        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);

        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * ne11 * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * ne11 * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * ne11 * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * ne11 * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_FLASH_FF: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * node->src[1]->ne[1] * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * node->src[1]->ne[1] * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * node->src[1]->ne[1] * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * node->src[1]->ne[1] * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_FLASH_ATTN_BACK: {
        n_tasks = n_threads;

        size_t cur = 0;

        const int64_t D = node->src[0]->ne[0];
        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
        const int64_t mxDn = MAX(D, ne11) * 2;  // *2 because of S and SM in ggml_compute_forward_flash_attn_back
        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * mxDn * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * mxDn * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * mxDn * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * mxDn * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_WIN_PART:
      case GGML_OP_WIN_UNPART:
      case GGML_OP_MAP_UNARY:
      case GGML_OP_MAP_BINARY:
      case GGML_OP_MAP_CUSTOM1:
      case GGML_OP_MAP_CUSTOM2:
      case GGML_OP_MAP_CUSTOM3: {
        n_tasks = 1;
      } break;
      case GGML_OP_CROSS_ENTROPY_LOSS: {
        n_tasks = n_threads;

        size_t cur = ggml_type_size(node->type) * (n_tasks + node->src[0]->ne[0] * n_tasks);

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
        n_tasks = n_threads;

        size_t cur = ggml_type_size(node->type) * node->src[0]->ne[0] * n_tasks;

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_NONE: {
        n_tasks = 1;
      } break;
      case GGML_OP_COUNT: {
      } break;
    }

    cplan.n_tasks[i] = n_tasks;
  }

  if (work_size > 0) {
    work_size += CACHE_LINE_SIZE * (n_threads - 1);
  }

  cplan.n_threads = n_threads;
  cplan.work_size = work_size;
  cplan.work_data = NULL;

  return cplan;
}

int ggml_graph_compute(struct ggml_cgraph *cgraph, struct ggml_cplan *cplan) {
  {
    if (cplan->work_size > 0) {
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
      if (cgraph->nodes[i]->op != GGML_OP_NONE) {
      }
    }
  }

  const int n_threads = cplan->n_threads;

  struct ggml_compute_state_shared state_shared = {
      /*.cgraph                  =*/cgraph,
      /*.cgraph_plan             =*/cplan,
      /*.perf_node_start_cycles  =*/0,
      /*.perf_node_start_time_us =*/0,
      /*.n_threads               =*/n_threads,
      /*.n_active                =*/n_threads,
      /*.node_n                  =*/-1,
      /*.abort_callback          =*/NULL,
      /*.abort_callback_data     =*/NULL,
  };
  struct ggml_compute_state *workers = alloca(sizeof(struct ggml_compute_state) * n_threads);

  // create thread pool
  if (n_threads > 1) {
    for (int j = 1; j < n_threads; ++j) {
      workers[j] = (struct ggml_compute_state){
          .thrd = 0,
          .ith = j,
          .shared = &state_shared,
      };

      ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
    }
  }
  workers[0].ith = 0;
  workers[0].shared = &state_shared;

  const int64_t perf_start_cycles = ggml_perf_cycles();
  const int64_t perf_start_time_us = ggml_perf_time_us();

  // this is a work thread too
  int compute_status = (size_t)ggml_graph_compute_thread(&workers[0]);

  // don't leave affinity set on the main thread
  clear_numa_thread_affinity();

  // join or kill thread pool
  if (n_threads > 1) {
    for (int j = 1; j < n_threads; j++) {
      ggml_thread_join(workers[j].thrd, NULL);
    }
  }

  // performance stats (graph)
  {
    int64_t perf_cycles_cur = ggml_perf_cycles() - perf_start_cycles;
    int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

    cgraph->perf_runs++;
    cgraph->perf_cycles += perf_cycles_cur;
    cgraph->perf_time_us += perf_time_us_cur;

    GGML_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n", __func__, cgraph->perf_runs,
                     (double)perf_cycles_cur / (double)ggml_cycles_per_ms(),
                     (double)cgraph->perf_cycles / (double)ggml_cycles_per_ms() / (double)cgraph->perf_runs,
                     (double)perf_time_us_cur / 1000.0, (double)cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
  }

  return compute_status;
}

void ggml_graph_reset(struct ggml_cgraph *cgraph) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *grad = cgraph->grads[i];

    if (grad) {
      ggml_set_zero(grad);
    }
  }
}

void ggml_graph_compute_with_ctx(struct ggml_context *ctx, struct ggml_cgraph *cgraph, int n_threads) {
  struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads);

  struct ggml_tensor *buf = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cplan.work_size);

  cplan.work_data = buf->data;

  ggml_graph_compute(cgraph, &cplan);
}

struct ggml_tensor *ggml_graph_get_tensor(struct ggml_cgraph *cgraph, const char *name) {
  for (int i = 0; i < cgraph->n_leafs; i++) {
    struct ggml_tensor *leaf = cgraph->leafs[i];

    if (strcmp(leaf->name, name) == 0) {
      return leaf;
    }
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *node = cgraph->nodes[i];

    if (strcmp(node->name, name) == 0) {
      return node;
    }
  }

  return NULL;
}

static void ggml_graph_export_leaf(const struct ggml_tensor *tensor, FILE *fout) {
  const int64_t *ne = tensor->ne;
  const size_t *nb = tensor->nb;

  fprintf(fout, "%-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
          ggml_type_name(tensor->type), ggml_op_name(tensor->op), tensor->n_dims, ne[0], ne[1], ne[2], ne[3], nb[0],
          nb[1], nb[2], nb[3], tensor->data, tensor->name);
}

static void ggml_graph_export_node(const struct ggml_tensor *tensor, const char *arg, FILE *fout) {
  const int64_t *ne = tensor->ne;
  const size_t *nb = tensor->nb;

  fprintf(fout,
          "%-6s %-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
          arg, ggml_type_name(tensor->type), ggml_op_name(tensor->op), tensor->n_dims, ne[0], ne[1], ne[2], ne[3],
          nb[0], nb[1], nb[2], nb[3], tensor->data, tensor->name);
}

void ggml_graph_export(const struct ggml_cgraph *cgraph, const char *fname) {
  uint64_t size_eval = 0;

  // compute size of intermediate results
  // TODO: does not take into account scratch buffers !!!!
  for (int i = 0; i < cgraph->n_nodes; ++i) {
    size_eval += ggml_nbytes(cgraph->nodes[i]);
  }

  // print
  {
    FILE *fout = stdout;

    fprintf(fout, "\n");
    fprintf(fout, "%-16s %8x\n", "magic", GGML_FILE_MAGIC);
    fprintf(fout, "%-16s %8d\n", "version", GGML_FILE_VERSION);
    fprintf(fout, "%-16s %8d\n", "leafs", cgraph->n_leafs);
    fprintf(fout, "%-16s %8d\n", "nodes", cgraph->n_nodes);
    fprintf(fout, "%-16s %" PRIu64 "\n", "eval", size_eval);

    // header
    fprintf(fout, "\n");
    fprintf(fout, "%-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %16s %16s\n", "TYPE", "OP", "NDIMS", "NE0", "NE1",
            "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "NAME");

    for (int i = 0; i < cgraph->n_leafs; ++i) {
      ggml_graph_export_leaf(cgraph->leafs[i], fout);
    }

    // header
    fprintf(fout, "\n");
    fprintf(fout, "%-6s %-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %8s %16s %16s\n", "ARG", "TYPE", "OP",
            "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "NTASKS", "DATA", "NAME");

    for (int i = 0; i < cgraph->n_nodes; ++i) {
      ggml_graph_export_node(cgraph->nodes[i], "DST", fout);

      for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (cgraph->nodes[i]->src[j]) {
          ggml_graph_export_node(cgraph->nodes[i]->src[j], "SRC", fout);
        }
      }

      fprintf(fout, "\n");
    }

    fprintf(fout, "\n");
  }

  // write binary data
  {
    FILE *fout = fopen(fname, "wb");

    if (!fout) {
      fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
      return;
    }

    // header
    {
      const uint32_t magic = GGML_FILE_MAGIC;
      const uint32_t version = GGML_FILE_VERSION;
      const uint32_t n_leafs = cgraph->n_leafs;
      const uint32_t nodes = cgraph->n_nodes;

      fwrite(&magic, sizeof(uint32_t), 1, fout);
      fwrite(&version, sizeof(uint32_t), 1, fout);
      fwrite(&n_leafs, sizeof(uint32_t), 1, fout);
      fwrite(&nodes, sizeof(uint32_t), 1, fout);
      fwrite(&size_eval, sizeof(uint64_t), 1, fout);
    }

    // leafs
    {
      for (int i = 0; i < cgraph->n_leafs; ++i) {
        const struct ggml_tensor *tensor = cgraph->leafs[i];

        const uint32_t type = tensor->type;
        const uint32_t op = tensor->op;
        const uint32_t n_dims = tensor->n_dims;

        fwrite(&type, sizeof(uint32_t), 1, fout);
        fwrite(&op, sizeof(uint32_t), 1, fout);
        fwrite(&n_dims, sizeof(uint32_t), 1, fout);

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          const uint64_t ne = tensor->ne[j];
          const uint64_t nb = tensor->nb[j];

          fwrite(&ne, sizeof(uint64_t), 1, fout);
          fwrite(&nb, sizeof(uint64_t), 1, fout);
        }

        fwrite(tensor->name, sizeof(char), GGML_MAX_NAME, fout);

        // dump the data
        // TODO: pad this to 32 byte boundary
        {
          const size_t size = ggml_nbytes(tensor);

          fwrite(tensor->data, sizeof(char), size, fout);
        }
      }
    }

    // nodes
    {
      for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor *tensor = cgraph->nodes[i];

        const uint32_t type = tensor->type;
        const uint32_t op = tensor->op;
        const uint32_t n_dims = tensor->n_dims;

        fwrite(&type, sizeof(uint32_t), 1, fout);
        fwrite(&op, sizeof(uint32_t), 1, fout);
        fwrite(&n_dims, sizeof(uint32_t), 1, fout);

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          const uint64_t ne = tensor->ne[j];
          const uint64_t nb = tensor->nb[j];

          fwrite(&ne, sizeof(uint64_t), 1, fout);
          fwrite(&nb, sizeof(uint64_t), 1, fout);
        }

        fwrite(tensor->name, sizeof(char), GGML_MAX_NAME, fout);

        // output the op arguments
        {
          struct ggml_tensor *args[GGML_MAX_SRC] = {NULL};

          for (int j = 0; j < GGML_MAX_SRC; ++j) {
            args[j] = tensor->src[j];
          }

          for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (args[j]) {
              int32_t idx = -1;

              // check if leaf
              {
                for (int k = 0; k < cgraph->n_leafs; ++k) {
                  if (args[j] == cgraph->leafs[k]) {
                    idx = k;
                    break;
                  }
                }
              }

              // check if node
              if (idx == -1) {
                for (int k = 0; k < cgraph->n_nodes; ++k) {
                  if (args[j] == cgraph->nodes[k]) {
                    idx = GGML_MAX_NODES + k;
                    break;
                  }
                }
              }

              if (idx == -1) {
                fprintf(stderr, "%s: failed to find tensor, arg = %d, node = %d\n", __func__, j, i);
                return;
              }

              fwrite(&idx, sizeof(int32_t), 1, fout);
            } else {
              const int32_t nul = -1;

              fwrite(&nul, sizeof(int32_t), 1, fout);
            }
          }
        }
      }
    }

    fclose(fout);
  }
}

struct ggml_cgraph ggml_graph_import(const char *fname, struct ggml_context **ctx_data,
                                     struct ggml_context **ctx_eval) {
  struct ggml_cgraph result = {0};

  struct ggml_tensor *data = NULL;

  // read file into data
  {
    FILE *fin = fopen(fname, "rb");
    if (!fin) {
      fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
      return result;
    }

    size_t fsize = 0;

    fseek(fin, 0, SEEK_END);
    fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    // create the data context
    {
      const size_t overhead = 1 * ggml_tensor_overhead();

      struct ggml_init_params params = {
          .mem_size = fsize + overhead,
          .mem_buffer = NULL,
          .no_alloc = false,
      };

      *ctx_data = ggml_init(params);

      if (!*ctx_data) {
        fprintf(stderr, "%s: failed to create ggml context\n", __func__);
        fclose(fin);
        return result;
      }
    }

    data = ggml_new_tensor_1d(*ctx_data, GGML_TYPE_I8, fsize);

    {
      const size_t ret = fread(data->data, sizeof(char), fsize, fin);
      if (ret != fsize) {
        fprintf(stderr, "%s: failed to read %s\n", __func__, fname);
        fclose(fin);
        return result;
      }
    }

    fclose(fin);
  }

  // populate result
  {
    char *ptr = (char *)data->data;

    const uint32_t magic = *(const uint32_t *)ptr;
    ptr += sizeof(magic);

    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid magic number, got %08x\n", __func__, magic);
      return result;
    }

    const uint32_t version = *(const uint32_t *)ptr;
    ptr += sizeof(version);

    if (version != GGML_FILE_VERSION) {
      fprintf(stderr, "%s: invalid version number\n", __func__);
      return result;
    }

    const uint32_t n_leafs = *(const uint32_t *)ptr;
    ptr += sizeof(n_leafs);
    const uint32_t n_nodes = *(const uint32_t *)ptr;
    ptr += sizeof(n_nodes);
    const uint64_t size_eval = *(const uint64_t *)ptr;
    ptr += sizeof(size_eval);

    result.n_leafs = n_leafs;
    result.n_nodes = n_nodes;

    // create the data context
    {
      const size_t overhead = (n_leafs + n_nodes) * ggml_tensor_overhead();

      struct ggml_init_params params = {
          .mem_size = size_eval + overhead,
          .mem_buffer = NULL,
          .no_alloc = true,
      };

      *ctx_eval = ggml_init(params);

      if (!*ctx_eval) {
        fprintf(stderr, "%s: failed to create ggml context\n", __func__);
        return result;
      }
    }

    // leafs
    {
      uint32_t type;
      uint32_t op;
      uint32_t n_dims;

      for (uint32_t i = 0; i < n_leafs; ++i) {
        type = *(const uint32_t *)ptr;
        ptr += sizeof(type);
        op = *(const uint32_t *)ptr;
        ptr += sizeof(op);
        n_dims = *(const uint32_t *)ptr;
        ptr += sizeof(n_dims);

        int64_t ne[GGML_MAX_DIMS];
        size_t nb[GGML_MAX_DIMS];

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          uint64_t ne_cur;
          uint64_t nb_cur;

          ne_cur = *(const uint64_t *)ptr;
          ptr += sizeof(ne_cur);
          nb_cur = *(const uint64_t *)ptr;
          ptr += sizeof(nb_cur);

          ne[j] = ne_cur;
          nb[j] = nb_cur;
        }

        struct ggml_tensor *tensor = ggml_new_tensor(*ctx_eval, (enum ggml_type)type, n_dims, ne);

        tensor->op = (enum ggml_op)op;

        memcpy(tensor->name, ptr, GGML_MAX_NAME);
        ptr += GGML_MAX_NAME;

        tensor->data = (void *)ptr;

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          tensor->nb[j] = nb[j];
        }

        result.leafs[i] = tensor;

        ptr += ggml_nbytes(tensor);

        fprintf(stderr, "%s: loaded leaf %d: '%16s', %3d dims, %9zu bytes\n", __func__, i, tensor->name, n_dims,
                ggml_nbytes(tensor));
      }
    }

    ggml_set_no_alloc(*ctx_eval, false);

    // nodes
    {
      uint32_t type;
      uint32_t op;
      uint32_t n_dims;

      for (uint32_t i = 0; i < n_nodes; ++i) {
        type = *(const uint32_t *)ptr;
        ptr += sizeof(type);
        op = *(const uint32_t *)ptr;
        ptr += sizeof(op);
        n_dims = *(const uint32_t *)ptr;
        ptr += sizeof(n_dims);

        enum ggml_op eop = (enum ggml_op)op;

        int64_t ne[GGML_MAX_DIMS];
        size_t nb[GGML_MAX_DIMS];

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          uint64_t ne_cur;
          uint64_t nb_cur;

          ne_cur = *(const uint64_t *)ptr;
          ptr += sizeof(ne_cur);
          nb_cur = *(const uint64_t *)ptr;
          ptr += sizeof(nb_cur);

          ne[j] = ne_cur;
          nb[j] = nb_cur;
        }

        const char *ptr_name = ptr;
        ptr += GGML_MAX_NAME;

        const int32_t *ptr_arg_idx = (const int32_t *)ptr;
        ptr += GGML_MAX_SRC * sizeof(int32_t);

        struct ggml_tensor *args[GGML_MAX_SRC] = {NULL};

        // parse args
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
          const int32_t arg_idx = ptr_arg_idx[j];

          if (arg_idx == -1) {
            continue;
          }

          if (arg_idx < GGML_MAX_NODES) {
            args[j] = result.leafs[arg_idx];
          } else {
            args[j] = result.nodes[arg_idx - GGML_MAX_NODES];
          }
        }

        // create the tensor
        // "view" operations are handled differently
        // TODO: handle inplace ops - currently a copy is always made

        struct ggml_tensor *tensor = NULL;

        switch (eop) {
          // TODO: implement other view ops
          case GGML_OP_RESHAPE: {
            tensor = ggml_reshape_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3]);
          } break;
          case GGML_OP_VIEW: {
            tensor = ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);

            uint64_t offs;
            memcpy(&offs, args[2]->data, sizeof(offs));

            tensor->data = ((char *)tensor->data) + offs;
          } break;
          case GGML_OP_TRANSPOSE: {
            tensor = ggml_transpose(*ctx_eval, args[0]);
          } break;
          case GGML_OP_PERMUTE: {
            tensor = ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);
          } break;
          default: {
            tensor = ggml_new_tensor(*ctx_eval, (enum ggml_type)type, n_dims, ne);

            tensor->op = eop;
          } break;
        }

        memcpy(tensor->name, ptr_name, GGML_MAX_NAME);

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
          tensor->nb[j] = nb[j];
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
          tensor->src[j] = args[j];
        }

        result.nodes[i] = tensor;

        fprintf(stderr, "%s: loaded node %d: '%16s', %3d dims, %9zu bytes\n", __func__, i, tensor->name, n_dims,
                ggml_nbytes(tensor));
      }
    }
  }

  return result;
}

void ggml_graph_print(const struct ggml_cgraph *cgraph) {
  int64_t perf_total_per_op_us[GGML_OP_COUNT] = {0};

  printf("=== GRAPH ===\n");

  GGML_PRINT_DEBUG("n_threads       = %d\n", cgraph->n_threads);
  GGML_PRINT_DEBUG("total work size = %zu bytes\n", cgraph->work_size);

  printf("n_nodes = %d\n", cgraph->n_nodes);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *node = cgraph->nodes[i];

    perf_total_per_op_us[node->op] += MAX(1, node->perf_time_us);

    printf(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64
           "] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
           i, node->ne[0], node->ne[1], node->ne[2], GGML_OP_NAME[node->op],
           node->is_param ? "x"
           : node->grad   ? "g"
                          : " ",
           node->perf_runs, (double)node->perf_cycles / (double)ggml_cycles_per_ms(),
           (double)node->perf_cycles / (double)ggml_cycles_per_ms() / (double)node->perf_runs,
           (double)node->perf_time_us / 1000.0, (double)node->perf_time_us / 1000.0 / node->perf_runs);
  }

  printf("n_leafs = %d\n", cgraph->n_leafs);
  for (int i = 0; i < cgraph->n_leafs; i++) {
    struct ggml_tensor *node = cgraph->leafs[i];

    printf(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s\n", i, node->ne[0], node->ne[1], GGML_OP_NAME[node->op]);
  }

  for (int i = 0; i < GGML_OP_COUNT; i++) {
    if (perf_total_per_op_us[i] == 0) {
      continue;
    }

    printf("perf_total_per_op_us[%16s] = %7.3f ms\n", GGML_OP_NAME[i], (double)perf_total_per_op_us[i] / 1000.0);
  }

  printf("========================================\n");
}

// check if node is part of the graph
static bool ggml_graph_find(const struct ggml_cgraph *cgraph, const struct ggml_tensor *node) {
  if (cgraph == NULL) {
    return true;
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return true;
    }
  }

  return false;
}

static struct ggml_tensor *ggml_graph_get_parent(const struct ggml_cgraph *cgraph, const struct ggml_tensor *node) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor *parent = cgraph->nodes[i];

    if (parent->grad == node) {
      return parent;
    }
  }

  return NULL;
}

static void ggml_graph_dump_dot_node_edge(FILE *fp, const struct ggml_cgraph *gb, struct ggml_tensor *node,
                                          struct ggml_tensor *parent, const char *label) {
  struct ggml_tensor *gparent = ggml_graph_get_parent(gb, node);
  struct ggml_tensor *gparent0 = ggml_graph_get_parent(gb, parent);
  fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
          gparent0 ? (void *)gparent0 : (void *)parent, gparent0 ? "g" : "x", gparent ? (void *)gparent : (void *)node,
          gparent ? "g" : "x", gparent ? "empty" : "vee", gparent ? "dashed" : "solid", label);
}

static void ggml_graph_dump_dot_leaf_edge(FILE *fp, struct ggml_tensor *node, struct ggml_tensor *parent,
                                          const char *label) {
  fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"%s\"; ]\n", (void *)parent, "x", (void *)node, "x", label);
}

void ggml_graph_dump_dot(const struct ggml_cgraph *gb, const struct ggml_cgraph *gf, const char *filename) {
  char color[16];

  FILE *fp = fopen(filename, "w");

  fprintf(fp, "digraph G {\n");
  fprintf(fp, "  newrank = true;\n");
  fprintf(fp, "  rankdir = LR;\n");

  for (int i = 0; i < gb->n_nodes; i++) {
    struct ggml_tensor *node = gb->nodes[i];

    if (ggml_graph_get_parent(gb, node) != NULL) {
      continue;
    }

    if (node->is_param) {
      snprintf(color, sizeof(color), "yellow");
    } else if (node->grad) {
      if (ggml_graph_find(gf, node)) {
        snprintf(color, sizeof(color), "green");
      } else {
        snprintf(color, sizeof(color), "lightblue");
      }
    } else {
      snprintf(color, sizeof(color), "white");
    }

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"",
            (void *)node, color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s (%s)|", node->name, ggml_type_name(node->type));
    } else {
      fprintf(fp, "(%s)|", ggml_type_name(node->type));
    }

    if (node->n_dims == 2) {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], GGML_OP_SYMBOL[node->op]);
    } else {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], node->ne[2],
              GGML_OP_SYMBOL[node->op]);
    }

    if (node->grad) {
      fprintf(fp, " | <g>%s\"; ]\n", GGML_OP_SYMBOL[node->grad->op]);
    } else {
      fprintf(fp, "\"; ]\n");
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    struct ggml_tensor *node = gb->leafs[i];

    snprintf(color, sizeof(color), "pink");

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"<x>",
            (void *)node, color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s (%s)|", node->name, ggml_type_name(node->type));
    } else {
      fprintf(fp, "(%s)|", ggml_type_name(node->type));
    }

    fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
    if (ggml_nelements(node) < 5) {
      fprintf(fp, " | (");
      for (int j = 0; j < ggml_nelements(node); j++) {
        if (node->type == GGML_TYPE_I8 || node->type == GGML_TYPE_I16 || node->type == GGML_TYPE_I32) {
          fprintf(fp, "%d", ggml_get_i32_1d(node, j));
        } else if (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16) {
          fprintf(fp, "%.1e", (double)ggml_get_f32_1d(node, j));
        } else {
          fprintf(fp, "#");
        }
        if (j < ggml_nelements(node) - 1) {
          fprintf(fp, ", ");
        }
      }
      fprintf(fp, ")");
    }
    fprintf(fp, "\"; ]\n");
  }

  for (int i = 0; i < gb->n_nodes; i++) {
    struct ggml_tensor *node = gb->nodes[i];

    for (int j = 0; j < GGML_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        ggml_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
      }
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    struct ggml_tensor *node = gb->leafs[i];

    for (int j = 0; j < GGML_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        ggml_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
      }
    }
  }

  fprintf(fp, "}\n");

  fclose(fp);

  printf("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void ggml_opt_set_params(int np, struct ggml_tensor *const ps[], const float *x) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ggml_nelements(ps[p]);
    // TODO: add function to set tensor from array
    for (int64_t j = 0; j < ne; ++j) {
      ggml_set_f32_1d(ps[p], j, x[i++]);
    }
  }
}

static void ggml_opt_get_params(int np, struct ggml_tensor *const ps[], float *x) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ggml_nelements(ps[p]);
    // TODO: add function to get all elements at once
    for (int64_t j = 0; j < ne; ++j) {
      x[i++] = ggml_get_f32_1d(ps[p], j);
    }
  }
}

static void ggml_opt_get_grad(int np, struct ggml_tensor *const ps[], float *g) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ggml_nelements(ps[p]);
    // TODO: add function to get all elements at once
    for (int64_t j = 0; j < ne; ++j) {
      g[i++] = ggml_get_f32_1d(ps[p]->grad, j);
    }
  }
}

//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

static enum ggml_opt_result ggml_opt_adam(struct ggml_context *ctx, struct ggml_opt_context *opt,
                                          struct ggml_opt_params params, struct ggml_tensor *f, struct ggml_cgraph *gf,
                                          struct ggml_cgraph *gb) {
  // these will store the parameters we want to optimize
  struct ggml_tensor *ps[GGML_MAX_PARAMS];

  int np = 0;
  int nx = 0;
  for (int i = 0; i < gf->n_nodes; ++i) {
    if (gf->nodes[i]->is_param) {
      GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

      ps[np++] = gf->nodes[i];
      nx += ggml_nelements(gf->nodes[i]);
    }
  }

  if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past)) {
    int iter = opt->iter;
    ggml_opt_init(opt->ctx, opt, params, nx);
    opt->iter = iter;
  }

  // constants
  const float sched = params.adam.sched;
  const float decay = params.adam.decay * sched;
  const float alpha = params.adam.alpha * sched;
  const float beta1 = params.adam.beta1;
  const float beta2 = params.adam.beta2;
  const float eps = params.adam.eps;

  float *x = opt->adam.x->data;    // view of the parameters
  float *g1 = opt->adam.g1->data;  // gradient
  float *g2 = opt->adam.g2->data;  // gradient squared
  float *m = opt->adam.m->data;    // first moment
  float *v = opt->adam.v->data;    // second moment
  float *mh = opt->adam.mh->data;  // first moment hat
  float *vh = opt->adam.vh->data;  // second moment hat

  float *pf = params.past > 0 ? opt->adam.pf->data : NULL;  // past function values

  // update view
  ggml_opt_get_params(np, ps, x);

  // compute the function value
  ggml_graph_reset(gf);
  ggml_set_f32(f->grad, 1.0f);

  ggml_graph_compute_with_ctx(ctx, gb, params.n_threads);

  opt->adam.fx_prev = ggml_get_f32_1d(f, 0);
  opt->adam.fx_best = opt->adam.fx_prev;
  if (pf) {
    pf[opt->iter % params.past] = opt->adam.fx_prev;
  }

  // initialize
  if (opt->just_initialized) {
    opt->adam.n_no_improvement = 0;
    opt->just_initialized = false;
  }

  float *fx_best = &opt->adam.fx_best;
  float *fx_prev = &opt->adam.fx_prev;
  int *n_no_improvement = &opt->adam.n_no_improvement;

  int iter0 = opt->iter;

  // run the optimizer
  for (int t = 0; t < params.adam.n_iter; ++t) {
    opt->iter = iter0 + t + 1;
    GGML_PRINT_DEBUG("=== iter %d ===\n", t);

    GGML_PRINT_DEBUG("f      = %10.6f\n", ggml_get_f32_1d(f, 0));
    GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", ggml_get_f32_1d(ps[0]->grad, 0));
    GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", ggml_get_f32_1d(ps[1]->grad, 0));

    for (int i = 0; i < np; ++i) {
      GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i, ggml_get_f32_1d(ps[i], 0), ggml_get_f32_1d(ps[i]->grad, 0));
    }

    const int64_t t_start_wall = ggml_time_us();
    const int64_t t_start_cpu = ggml_cycles();
    UNUSED(t_start_wall);
    UNUSED(t_start_cpu);

    {
      // update the gradient
      ggml_opt_get_grad(np, ps, g1);

      // m_t = beta1*m_t-1 + (1 - beta1)*g_t
      ggml_vec_scale_f32(nx, m, beta1);
      ggml_vec_mad_f32(nx, m, g1, 1.0f - beta1);

      // g2 = g1^2
      ggml_vec_sqr_f32(nx, g2, g1);

      // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
      ggml_vec_scale_f32(nx, v, beta2);
      ggml_vec_mad_f32(nx, v, g2, 1.0f - beta2);

      // m^hat = m_t / (1 - beta1^t)
      // v^hat = v_t / (1 - beta2^t)
      // x_t = x_t-1 - sched*(alpha*m^hat/(sqrt(v^hat) + eps) + decay*x_t-1)
      // x_t = x_t-1 - sched*alpha*m^hat/(sqrt(v^hat) + eps) - sched*decay*x_t-1
      // x_t = x_t-1*(1-sched*decay) - sched*alpha*m^hat/(sqrt(v^hat) + eps)
      // x_t = x_t-1*(1-sched*decay) + sched*decay*(-alpha/decay)*m^hat/(sqrt(v^hat) + eps)
      // x_t = mix(x_t-1, (-alpha/decay)*m^hat/(sqrt(v^hat) + eps), sched*decay)
      ggml_vec_cpy_f32(nx, mh, m);
      ggml_vec_cpy_f32(nx, vh, v);

      ggml_vec_scale_f32(nx, mh, alpha / (1.0f - powf(beta1, opt->iter)));
      ggml_vec_scale_f32(nx, vh, 1.0f / (1.0f - powf(beta2, opt->iter)));

      ggml_vec_sqrt_f32(nx, vh, vh);
      ggml_vec_acc1_f32(nx, vh, eps);

      ggml_vec_div_f32(nx, mh, mh, vh);
      ggml_vec_scale_f32(nx, x, 1.0f - decay);
      ggml_vec_sub_f32(nx, x, x, mh);

      // update the parameters
      ggml_opt_set_params(np, ps, x);
    }

    ggml_graph_reset(gf);
    ggml_set_f32(f->grad, 1.0f);

    ggml_graph_compute_with_ctx(ctx, gb, params.n_threads);

    const float fx = ggml_get_f32_1d(f, 0);

    // check convergence
    if (fabsf(fx - fx_prev[0]) / fx < params.adam.eps_f) {
      GGML_PRINT_DEBUG("converged\n");

      return GGML_OPT_OK;
    }

    // delta-based convergence test
    if (pf != NULL) {
      // need at least params.past iterations to start checking for convergence
      if (params.past <= iter0 + t) {
        const float rate = (pf[(iter0 + t) % params.past] - fx) / fx;

        if (fabsf(rate) < params.delta) {
          return GGML_OPT_OK;
        }
      }

      pf[(iter0 + t) % params.past] = fx;
    }

    // check for improvement
    if (params.max_no_improvement > 0) {
      if (fx_best[0] > fx) {
        fx_best[0] = fx;
        n_no_improvement[0] = 0;
      } else {
        ++n_no_improvement[0];

        if (n_no_improvement[0] >= params.max_no_improvement) {
          return GGML_OPT_OK;
        }
      }
    }

    fx_prev[0] = fx;

    {
      const int64_t t_end_cpu = ggml_cycles();
      GGML_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu)) / CLOCKS_PER_SEC);
      UNUSED(t_end_cpu);

      const int64_t t_end_wall = ggml_time_us();
      GGML_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall) / 1e6);
      UNUSED(t_end_wall);
    }
  }

  return GGML_OPT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct ggml_lbfgs_iteration_data {
  float alpha;
  float ys;
  float *s;
  float *y;
};

static enum ggml_opt_result linesearch_backtracking(struct ggml_context *ctx, const struct ggml_opt_params *params,
                                                    int nx, float *x, float *fx, float *g, float *d, float *step,
                                                    const float *xp, struct ggml_tensor *f, struct ggml_cgraph *gf,
                                                    struct ggml_cgraph *gb, const int np, struct ggml_tensor *ps[]) {
  int count = 0;

  float width = 0.0f;
  float dg = 0.0f;
  float finit = 0.0f;
  float dginit = 0.0f;
  float dgtest = 0.0f;

  const float dec = 0.5f;
  const float inc = 2.1f;

  if (*step <= 0.f) {
    return GGML_LINESEARCH_INVALID_PARAMETERS;
  }

  // compute the initial gradient in the search direction
  ggml_vec_dot_f32(nx, &dginit, g, d);

  // make sure that d points to a descent direction
  if (0 < dginit) {
    return GGML_LINESEARCH_FAIL;
  }

  // initialize local variables
  finit = *fx;
  dgtest = params->lbfgs.ftol * dginit;

  while (true) {
    ggml_vec_cpy_f32(nx, x, xp);
    ggml_vec_mad_f32(nx, x, d, *step);

    // evaluate the function and gradient values
    {
      ggml_opt_set_params(np, ps, x);

      ggml_graph_reset(gf);
      ggml_set_f32(f->grad, 1.0f);

      ggml_graph_compute_with_ctx(ctx, gb, params->n_threads);

      ggml_opt_get_grad(np, ps, g);

      *fx = ggml_get_f32_1d(f, 0);
    }

    ++count;

    if (*fx > finit + (*step) * dgtest) {
      width = dec;
    } else {
      // Armijo condition is satisfied
      if (params->lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_ARMIJO) {
        return count;
      }

      ggml_vec_dot_f32(nx, &dg, g, d);

      // check the Wolfe condition
      if (dg < params->lbfgs.wolfe * dginit) {
        width = inc;
      } else {
        if (params->lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_WOLFE) {
          // regular Wolfe conditions
          return count;
        }

        if (dg > -params->lbfgs.wolfe * dginit) {
          width = dec;
        } else {
          // strong Wolfe condition (GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
          return count;
        }
        return count;
      }
    }

    if (*step < params->lbfgs.min_step) {
      return GGML_LINESEARCH_MINIMUM_STEP;
    }
    if (*step > params->lbfgs.max_step) {
      return GGML_LINESEARCH_MAXIMUM_STEP;
    }
    if (params->lbfgs.max_linesearch <= count) {
      return GGML_LINESEARCH_MAXIMUM_ITERATIONS;
    }

    (*step) *= width;
  }

  return GGML_LINESEARCH_FAIL;
}

static enum ggml_opt_result ggml_opt_lbfgs(struct ggml_context *ctx, struct ggml_opt_context *opt,
                                           struct ggml_opt_params params, struct ggml_tensor *f, struct ggml_cgraph *gf,
                                           struct ggml_cgraph *gb) {
  if (params.lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_WOLFE ||
      params.lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
    if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1.f <= params.lbfgs.wolfe) {
      return GGML_OPT_INVALID_WOLFE;
    }
  }

  const int m = params.lbfgs.m;

  // these will store the parameters we want to optimize
  struct ggml_tensor *ps[GGML_MAX_PARAMS];

  int np = 0;
  int nx = 0;
  for (int i = 0; i < gf->n_nodes; ++i) {
    if (gf->nodes[i]->is_param) {
      GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

      ps[np++] = gf->nodes[i];
      nx += ggml_nelements(gf->nodes[i]);
    }
  }

  if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past) ||
      (opt->params.lbfgs.m != params.lbfgs.m)) {
    int iter = opt->iter;
    ggml_opt_init(ctx, opt, params, nx);
    opt->iter = iter;
  }

  float *x = opt->lbfgs.x->data;    // current parameters
  float *xp = opt->lbfgs.xp->data;  // previous parameters
  float *g = opt->lbfgs.g->data;    // current gradient
  float *gp = opt->lbfgs.gp->data;  // previous gradient
  float *d = opt->lbfgs.d->data;    // search direction

  float *pf = params.past > 0 ? opt->lbfgs.pf->data : NULL;  // past function values

  float fx = 0.0f;     // cost function value
  float xnorm = 0.0f;  // ||x||
  float gnorm = 0.0f;  // ||g||

  // initialize x from the graph nodes
  ggml_opt_get_params(np, ps, x);

  // the L-BFGS memory
  float *lm_alpha = opt->lbfgs.lmal->data;
  float *lm_ys = opt->lbfgs.lmys->data;
  float *lm_s = opt->lbfgs.lms->data;
  float *lm_y = opt->lbfgs.lmy->data;

  // evaluate the function value and its gradient
  {
    ggml_opt_set_params(np, ps, x);

    ggml_graph_reset(gf);
    ggml_set_f32(f->grad, 1.0f);

    ggml_graph_compute_with_ctx(ctx, gb, params.n_threads);

    ggml_opt_get_grad(np, ps, g);

    fx = ggml_get_f32_1d(f, 0);
  }

  // search direction = -gradient
  ggml_vec_neg_f32(nx, d, g);

  // ||x||, ||g||
  ggml_vec_norm_f32(nx, &xnorm, x);
  ggml_vec_norm_f32(nx, &gnorm, g);

  if (xnorm < 1.0f) {
    xnorm = 1.0f;
  }

  // already optimized
  if (gnorm / xnorm <= params.lbfgs.eps) {
    return GGML_OPT_OK;
  }

  if (opt->just_initialized) {
    if (pf) {
      pf[0] = fx;
    }
    opt->lbfgs.fx_best = fx;

    // initial step
    ggml_vec_norm_inv_f32(nx, &opt->lbfgs.step, d);
    opt->lbfgs.j = 0;
    opt->lbfgs.k = 1;
    opt->lbfgs.end = 0;
    opt->lbfgs.n_no_improvement = 0;
    opt->just_initialized = false;
  }

  float *fx_best = &opt->lbfgs.fx_best;
  float *step = &opt->lbfgs.step;
  int *j = &opt->lbfgs.j;
  int *k = &opt->lbfgs.k;
  int *end = &opt->lbfgs.end;
  int *n_no_improvement = &opt->lbfgs.n_no_improvement;

  int ls = 0;
  int bound = 0;

  float ys = 0.0f;
  float yy = 0.0f;
  float beta = 0.0f;

  int it = 0;

  while (true) {
    // store the current position and gradient vectors
    ggml_vec_cpy_f32(nx, xp, x);
    ggml_vec_cpy_f32(nx, gp, g);

    ls = linesearch_backtracking(ctx, &params, nx, x, &fx, g, d, step, xp, f, gf, gb, np, ps);

    if (ls < 0) {
      // linesearch failed - go back to the previous point and return
      ggml_vec_cpy_f32(nx, x, xp);
      ggml_vec_cpy_f32(nx, g, gp);

      return ls;
    }

    ggml_vec_norm_f32(nx, &xnorm, x);
    ggml_vec_norm_f32(nx, &gnorm, g);

    GGML_PRINT_DEBUG("f = %10.6f\n", ggml_get_f32_1d(f, 0));

    if (xnorm < 1.0f) {
      xnorm = 1.0f;
    }
    if (gnorm / xnorm <= params.lbfgs.eps) {
      // converged
      return GGML_OPT_OK;
    }

    // delta-based convergence test
    if (pf != NULL) {
      // need at least params.past iterations to start checking for convergence
      if (params.past <= k[0]) {
        const float rate = (pf[k[0] % params.past] - fx) / fx;

        if (fabsf(rate) < params.delta) {
          return GGML_OPT_OK;
        }
      }

      pf[k[0] % params.past] = fx;
    }

    // check for improvement
    if (params.max_no_improvement > 0) {
      if (fx < fx_best[0]) {
        fx_best[0] = fx;
        n_no_improvement[0] = 0;
      } else {
        n_no_improvement[0]++;

        if (n_no_improvement[0] >= params.max_no_improvement) {
          return GGML_OPT_OK;
        }
      }
    }

    if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < it + 1) {
      // reached the maximum number of iterations
      return GGML_OPT_DID_NOT_CONVERGE;
    }

    // update vectors s and y:
    //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
    //   y_{k+1} = g_{k+1} - g_{k}.
    //
    ggml_vec_sub_f32(nx, &lm_s[end[0] * nx], x, xp);
    ggml_vec_sub_f32(nx, &lm_y[end[0] * nx], g, gp);

    // compute scalars ys and yy:
    //     ys = y^t \cdot s    -> 1 / \rho.
    //     yy = y^t \cdot y.
    //
    ggml_vec_dot_f32(nx, &ys, &lm_y[end[0] * nx], &lm_s[end[0] * nx]);
    ggml_vec_dot_f32(nx, &yy, &lm_y[end[0] * nx], &lm_y[end[0] * nx]);

    lm_ys[end[0]] = ys;

    // find new search direction
    //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

    bound = (m <= k[0]) ? m : k[0];
    k[0]++;
    it++;
    end[0] = (end[0] + 1) % m;

    // initialize search direction with -g
    ggml_vec_neg_f32(nx, d, g);

    j[0] = end[0];
    for (int i = 0; i < bound; ++i) {
      j[0] = (j[0] + m - 1) % m;
      // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
      ggml_vec_dot_f32(nx, &lm_alpha[j[0]], &lm_s[j[0] * nx], d);
      lm_alpha[j[0]] /= lm_ys[j[0]];
      // q_{i} = q_{i+1} - \alpha_{i} y_{i}
      ggml_vec_mad_f32(nx, d, &lm_y[j[0] * nx], -lm_alpha[j[0]]);
    }

    ggml_vec_scale_f32(nx, d, ys / yy);

    for (int i = 0; i < bound; ++i) {
      // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
      ggml_vec_dot_f32(nx, &beta, &lm_y[j[0] * nx], d);
      beta /= lm_ys[j[0]];
      // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
      ggml_vec_mad_f32(nx, d, &lm_s[j[0] * nx], lm_alpha[j[0]] - beta);
      j[0] = (j[0] + 1) % m;
    }

    step[0] = 1.0;
  }

  return GGML_OPT_DID_NOT_CONVERGE;
}

struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type) {
  struct ggml_opt_params result;

  switch (type) {
    case GGML_OPT_ADAM: {
      result = (struct ggml_opt_params){
          .type = GGML_OPT_ADAM,
          .n_threads = 1,
          .past = 0,
          .delta = 1e-5f,

          .max_no_improvement = 100,

          .print_forward_graph = true,
          .print_backward_graph = true,

          .adam =
              {
                  .n_iter = 10000,
                  .sched = 1.000f,
                  .decay = 0.001f,
                  .alpha = 0.001f,
                  .beta1 = 0.9f,
                  .beta2 = 0.999f,
                  .eps = 1e-8f,
                  .eps_f = 1e-5f,
                  .eps_g = 1e-3f,
              },
      };
    } break;
    case GGML_OPT_LBFGS: {
      result = (struct ggml_opt_params){
          .type = GGML_OPT_LBFGS,
          .n_threads = 1,
          .past = 0,
          .delta = 1e-5f,

          .max_no_improvement = 0,

          .print_forward_graph = true,
          .print_backward_graph = true,

          .lbfgs =
              {
                  .m = 6,
                  .n_iter = 100,
                  .max_linesearch = 20,

                  .eps = 1e-5f,
                  .ftol = 1e-4f,
                  .wolfe = 0.9f,
                  .min_step = 1e-20f,
                  .max_step = 1e+20f,

                  .linesearch = GGML_LINESEARCH_DEFAULT,
              },
      };
    } break;
  }

  return result;
}

void ggml_opt_init(struct ggml_context *ctx, struct ggml_opt_context *opt, struct ggml_opt_params params, int64_t nx) {
  opt->ctx = ctx;
  opt->params = params;
  opt->iter = 0;
  opt->nx = nx;
  opt->just_initialized = true;
  switch (opt->params.type) {
    case GGML_OPT_ADAM: {
      opt->adam.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.g2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.mh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.vh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->adam.pf = params.past > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.past) : NULL;
      ggml_set_zero(opt->adam.x);
      ggml_set_zero(opt->adam.g1);
      ggml_set_zero(opt->adam.g2);
      ggml_set_zero(opt->adam.m);
      ggml_set_zero(opt->adam.v);
      ggml_set_zero(opt->adam.mh);
      ggml_set_zero(opt->adam.vh);
      if (opt->adam.pf) {
        ggml_set_zero(opt->adam.pf);
      }
    } break;
    case GGML_OPT_LBFGS: {
      opt->lbfgs.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->lbfgs.xp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->lbfgs.g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->lbfgs.gp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->lbfgs.d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx);
      opt->lbfgs.pf = params.past > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.past) : NULL;
      opt->lbfgs.lmal = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.lbfgs.m);
      opt->lbfgs.lmys = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.lbfgs.m);
      opt->lbfgs.lms = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, params.lbfgs.m);
      opt->lbfgs.lmy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, params.lbfgs.m);
      ggml_set_zero(opt->lbfgs.x);
      ggml_set_zero(opt->lbfgs.xp);
      ggml_set_zero(opt->lbfgs.g);
      ggml_set_zero(opt->lbfgs.gp);
      ggml_set_zero(opt->lbfgs.d);
      if (opt->lbfgs.pf) {
        ggml_set_zero(opt->lbfgs.pf);
      }
      ggml_set_zero(opt->lbfgs.lmal);
      ggml_set_zero(opt->lbfgs.lmys);
      ggml_set_zero(opt->lbfgs.lms);
      ggml_set_zero(opt->lbfgs.lmy);
    } break;
  }
}

enum ggml_opt_result ggml_opt(struct ggml_context *ctx, struct ggml_opt_params params, struct ggml_tensor *f) {
  bool free_ctx = false;
  if (ctx == NULL) {
    struct ggml_init_params params_ctx = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    ctx = ggml_init(params_ctx);
    if (ctx == NULL) {
      return GGML_OPT_NO_CONTEXT;
    }

    free_ctx = true;
  }

  enum ggml_opt_result result = GGML_OPT_OK;

  struct ggml_opt_context *opt = (struct ggml_opt_context *)alloca(sizeof(struct ggml_opt_context));

  ggml_opt_init(ctx, opt, params, 0);
  result = ggml_opt_resume(ctx, opt, f);

  if (free_ctx) {
    ggml_free(ctx);
  }

  return result;
}

enum ggml_opt_result ggml_opt_resume(struct ggml_context *ctx, struct ggml_opt_context *opt, struct ggml_tensor *f) {
  // build forward + backward compute graphs
  struct ggml_tensor *gfbuf =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                         sizeof(struct ggml_cgraph) / GGML_TYPE_SIZE[GGML_TYPE_I32] +
                             (sizeof(struct ggml_cgraph) % GGML_TYPE_SIZE[GGML_TYPE_I32] ? 1 : 0));
  struct ggml_tensor *gbbuf =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                         sizeof(struct ggml_cgraph) / GGML_TYPE_SIZE[GGML_TYPE_I32] +
                             (sizeof(struct ggml_cgraph) % GGML_TYPE_SIZE[GGML_TYPE_I32] ? 1 : 0));

  struct ggml_cgraph *gf = (struct ggml_cgraph *)gfbuf->data;
  struct ggml_cgraph *gb = (struct ggml_cgraph *)gbbuf->data;

  *gf = ggml_build_forward(f);
  *gb = ggml_build_backward(ctx, gf, true);

  return ggml_opt_resume_g(ctx, opt, f, gf, gb);
}

enum ggml_opt_result ggml_opt_resume_g(struct ggml_context *ctx, struct ggml_opt_context *opt, struct ggml_tensor *f,
                                       struct ggml_cgraph *gf, struct ggml_cgraph *gb) {
  // build forward + backward compute graphs
  enum ggml_opt_result result = GGML_OPT_OK;

  switch (opt->params.type) {
    case GGML_OPT_ADAM: {
      result = ggml_opt_adam(ctx, opt, opt->params, f, gf, gb);
    } break;
    case GGML_OPT_LBFGS: {
      result = ggml_opt_lbfgs(ctx, opt, opt->params, f, gf, gb);
    } break;
  }

  if (opt->params.print_forward_graph) {
    ggml_graph_print(gf);
    ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
  }

  if (opt->params.print_backward_graph) {
    ggml_graph_print(gb);
    ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

size_t ggml_quantize_q4_0(const float *src, void *dst, int n, int k, int64_t *hist) {
  const int nb = k / QK4_0;

  for (int b = 0; b < n; b += k) {
    block_q4_0 *restrict y = (block_q4_0 *)dst + b / QK4_0;

    quantize_row_q4_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK4_0; j += 2) {
        const uint8_t vi0 = y[i].qs[j / 2] & 0x0F;
        const uint8_t vi1 = y[i].qs[j / 2] >> 4;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK4_0 * sizeof(block_q4_0));
}

size_t ggml_quantize_q4_1(const float *src, void *dst, int n, int k, int64_t *hist) {
  const int nb = k / QK4_1;

  for (int b = 0; b < n; b += k) {
    block_q4_1 *restrict y = (block_q4_1 *)dst + b / QK4_1;

    quantize_row_q4_1_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK4_1; j += 2) {
        const uint8_t vi0 = y[i].qs[j / 2] & 0x0F;
        const uint8_t vi1 = y[i].qs[j / 2] >> 4;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK4_1 * sizeof(block_q4_1));
}

size_t ggml_quantize_q5_0(const float *src, void *dst, int n, int k, int64_t *hist) {
  const int nb = k / QK5_0;

  for (int b = 0; b < n; b += k) {
    block_q5_0 *restrict y = (block_q5_0 *)dst + b / QK5_0;

    quantize_row_q5_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      uint32_t qh;
      memcpy(&qh, &y[i].qh, sizeof(qh));

      for (int j = 0; j < QK5_0; j += 2) {
        const uint8_t vh0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
        const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

        // cast to 16 bins
        const uint8_t vi0 = ((y[i].qs[j / 2] & 0x0F) | vh0) / 2;
        const uint8_t vi1 = ((y[i].qs[j / 2] >> 4) | vh1) / 2;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK5_0 * sizeof(block_q5_0));
}

size_t ggml_quantize_q5_1(const float *src, void *dst, int n, int k, int64_t *hist) {
  const int nb = k / QK5_1;

  for (int b = 0; b < n; b += k) {
    block_q5_1 *restrict y = (block_q5_1 *)dst + b / QK5_1;

    quantize_row_q5_1_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      uint32_t qh;
      memcpy(&qh, &y[i].qh, sizeof(qh));

      for (int j = 0; j < QK5_1; j += 2) {
        const uint8_t vh0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
        const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

        // cast to 16 bins
        const uint8_t vi0 = ((y[i].qs[j / 2] & 0x0F) | vh0) / 2;
        const uint8_t vi1 = ((y[i].qs[j / 2] >> 4) | vh1) / 2;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK5_1 * sizeof(block_q5_1));
}

size_t ggml_quantize_q8_0(const float *src, void *dst, int n, int k, int64_t *hist) {
  const int nb = k / QK8_0;

  for (int b = 0; b < n; b += k) {
    block_q8_0 *restrict y = (block_q8_0 *)dst + b / QK8_0;

    quantize_row_q8_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK8_0; ++j) {
        const int8_t vi = y[i].qs[j];

        hist[vi / 16 + 8]++;
      }
    }
  }

  return (n / QK8_0 * sizeof(block_q8_0));
}

size_t ggml_quantize_chunk(enum ggml_type type, const float *src, void *dst, int start, int n, int64_t *hist) {
  size_t result = 0;
  switch (type) {
    case GGML_TYPE_Q4_0: {
      block_q4_0 *block = (block_q4_0 *)dst + start / QK4_0;
      result = ggml_quantize_q4_0(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q4_1: {
      block_q4_1 *block = (block_q4_1 *)dst + start / QK4_1;
      result = ggml_quantize_q4_1(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q5_0: {
      block_q5_0 *block = (block_q5_0 *)dst + start / QK5_0;
      result = ggml_quantize_q5_0(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q5_1: {
      block_q5_1 *block = (block_q5_1 *)dst + start / QK5_1;
      result = ggml_quantize_q5_1(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q8_0: {
      block_q8_0 *block = (block_q8_0 *)dst + start / QK8_0;
      result = ggml_quantize_q8_0(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q2_K: {
      block_q2_K *block = (block_q2_K *)dst + start / QK_K;
      result = ggml_quantize_q2_K(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q3_K: {
      block_q3_K *block = (block_q3_K *)dst + start / QK_K;
      result = ggml_quantize_q3_K(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q4_K: {
      block_q4_K *block = (block_q4_K *)dst + start / QK_K;
      result = ggml_quantize_q4_K(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q5_K: {
      block_q5_K *block = (block_q5_K *)dst + start / QK_K;
      result = ggml_quantize_q5_K(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_Q6_K: {
      block_q6_K *block = (block_q6_K *)dst + start / QK_K;
      result = ggml_quantize_q6_K(src + start, block, n, n, hist);
    } break;
    case GGML_TYPE_F16: {
      int elemsize = sizeof(ggml_fp16_t);
      ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
      result = n * elemsize;
    } break;
    case GGML_TYPE_F32: {
      int elemsize = sizeof(float);
      result = n * elemsize;
      memcpy((uint8_t *)dst + start * elemsize, src + start, result);
    } break;
    default:
      abort();
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

int ggml_cpu_has_avx(void) { return 0; }

int ggml_cpu_has_avx2(void) { return 0; }

int ggml_cpu_has_avx512(void) { return 0; }

int ggml_cpu_has_avx512_vbmi(void) { return 0; }

int ggml_cpu_has_avx512_vnni(void) { return 0; }

int ggml_cpu_has_fma(void) { return 0; }

int ggml_cpu_has_neon(void) {
#if defined(__ARM_NEON)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_f16c(void) { return 0; }

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) { return 0; }

int ggml_cpu_has_blas(void) { return 1; }

int ggml_cpu_has_cublas(void) { return 0; }

int ggml_cpu_has_clblast(void) { return 0; }

int ggml_cpu_has_gpublas(void) { return ggml_cpu_has_cublas() || ggml_cpu_has_clblast(); }

int ggml_cpu_has_sse3(void) { return 0; }

int ggml_cpu_has_vsx(void) { return 0; }

////////////////////////////////////////////////////////////////////////////////
