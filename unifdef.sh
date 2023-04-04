defines=(
  -D__APPLE__
  -D__ARM_FEATURE_DOTPROD
  -D__ARM_FEATURE_FMA
  -D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  -D__ARM_FEATURE_QRDMX
  -D__ARM_NEON
  -DGGML_FP16_TO_FP32
  -DGGML_FP32_TO_FP16
  -DGGML_GELU_FP16
  -DGGML_MLOCK_SUPPORT
  -DGGML_SILU_FP16
  -DGGML_SIMD
  -DGGML_USE_ACCELERATE
  -DLLAMA_USE_SCRATCH
  -DNDEBUG
  -U __AVX512F
  -U__AVX__
  -U__AVX2__
  -U__AVX512F__
  -U__F16C__
  -U__FMA__
  -U__FreeBSD__
  -U__HAIKU__
  -U__MINGW32__
  -U__NetBSD__
  -U__OpenBSD__
  -U__POWER9_VECTOR__
  -U__SSE3__
  -U__wasm_simd128__
  -U_MSC_VER
  -U_WIN32
  -UGGML_DEBUG
  -UGGML_PERF
  -UGGML_SOFT_MAX_ACCELERATE
  -UGGML_USE_OPENBLAS
)

if [ "$1" == "-u" ]; then
  defines=("${defines[@]/-D/-U}")
fi

files=("ggml.c" "llama.cpp" "llama.h" "ggml.h")

for item in "${files[@]}"; do
  echo "Processing $item"
  git checkout master -- $item
  unifdef -k -o $item "${defines[@]}" $item
  clang-format -i $item
  git add $item
done
