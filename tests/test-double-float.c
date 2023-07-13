// These tests may take a long time!
// They are to prove that conversion from double to float of various functions in ggml.c doesn't affect the result.
// This is done by checking all finite (non-NaN, non-infinite) floats.

#undef NDEBUG
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

// ggml.c::quantize_row_q4_0_reference
inline static uint8_t round_orig(float v0) { return ((int8_t)(round(v0))) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_orig(float x) { return x / (1.0 + exp(-x)); }

#pragma GCC diagnostic pop

// ggml.c::quantize_row_q4_0_reference
inline static uint8_t round_float(float v0) { return (int8_t)roundf(v0) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_float(float x) { return x / (1.0f + expf(-x)); }

int main(void) {
  uint32_t x = UINT32_MAX;
  do {
    float f = *(float *)&x;
    assert(!isfinite(f) || (round_orig(f) == round_float(f)));
  } while (x--);
}
