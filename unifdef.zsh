#!/bin/zsh

ud() {
    unifdef -B -o $1 ${@:2} $1
}

defines=(
    -D __aarch64__
    -D __APPLE__
    -D __ARM_FEATURE_DOTPROD
    -D __ARM_FEATURE_FMA
    -D __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    -D __ARM_NEON
    -D __GNUC__
    -D __has_include
    -D __MACH__
    -D NDEBUG
    -k
    -U __AVX__
    -U __AVX2__
    -U __AVX2__
    -U __AVX512F__
    -U __AVX512VBMI__
    -U __AVX512VNNI__
    -U __AVXVNNI__
    -U __F16C__
    -U __FMA__
    -U __FreeBSD__
    -U __HAIKU__
    -U __linux__
    -U __MINGW32__
    -U __NetBSD__
    -U __OpenBSD__
    -U __POPCNT__
    -U __POWER9_VECTOR__
    -U __riscv
    -U __SSE3__
    -U __SSSE3__
    -U __unix__
    -U __wasm_simd128__
    -U _AIX
    -U _MSC_VER
    -U _WIN32
)

ggml_defines=(
    -D GGML_GELU_FP16
    -D GGML_GELU_QUICK_FP16
    -D GGML_METAL_NDEBUG
    -D GGML_SILU_FP16
    -D GGML_SIMD
    -D GGML_USE_ACCELERATE
    -D GGML_USE_K_QUANTS
    -D LLAMA_USE_SCRATCH
    -U GGML_PERF
    -U GGML_SHARED
    -U GGML_USE_CLBLAST
    -U GGML_USE_CUBLAS
    -U GGML_USE_MPI
    -U LLAMA_API_INTERNAL
    -U LLAMA_SHARED
)

for x in {*.c,*.cpp,*.h,*.m,examples/main/main.cpp,examples/common.h,examples/common.cpp}; do
    ud $x $defines
    ud $x $ggml_defines
    # clang-tidy --fix --fix-errors -p build/ $x
    clang-format --style=file -i $x
done
