#include <assert.h>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

const char *kernelSrc = R"(
#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;
typedef struct {
  float d;            // delta
  uint8_t qs[16]; // nibbles / quants
} block_q4_0;

kernel void add_arrays(constant const block_q4_0* x,
                       constant const block_q4_0* y,
                       device atomic_float* result,
                       uint2 gid [[thread_position_in_grid]])
{
    const int index = gid.x;
    const float deltaX = x[index].d;
    const float deltaY = y[index].d;

    constant const uint8_t* quantsX = x[index].qs;
    constant const uint8_t* quantsY = y[index].qs;

    float sumf = 0.0;

    for (int j = 0; j < 16; j++) {
      const uint8_t v0 = quantsX[j];
      const uint8_t v1 = quantsY[j];
      const int8_t v0_0 = (int8_t)(v0 & 0xf) - 8;
      const int8_t v0_1 = (int8_t)(v0 >> 4) - 8;
      const int8_t v1_0 = (int8_t)(v1 & 0xf) - 8;
      const int8_t v1_1 = (int8_t)(v1 >> 4) - 8;

      const float f0 = deltaX * v0_0;
      const float f1 = deltaX * v0_1;

      const float f2 = deltaY * v1_0;
      const float f3 = deltaY * v1_1;

      sumf += f0 * f2 + f1 * f3;
    }

    atomic_fetch_add_explicit(result, sumf, memory_order_relaxed);
}
)";

#define QK 32

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

static void ggml_vec_dot_q4_0(const int n, float *s, void *vx, void *vy) {
  const int nb = n / QK;

  const block_q4_0 *x = reinterpret_cast<block_q4_0 *>(vx);
  const block_q4_0 *y = reinterpret_cast<block_q4_0 *>(vy);

  float sumf = 0.0;

  // scalar
  for (int i = 0; i < nb; i++) {
    const float deltaX = x[i].d;
    const float deltaY = y[i].d;

    const uint8_t *quantsX = x[i].qs;
    const uint8_t *quantsY = y[i].qs;

    for (int j = 0; j < QK / 2; j++) {
      const uint8_t v0 = quantsX[j];
      const uint8_t v1 = quantsY[j];

      const float f0 = deltaX * ((int8_t)(v0 & 0xf) - 8);
      const float f1 = deltaX * ((int8_t)(v0 >> 4) - 8);

      const float f2 = deltaY * ((int8_t)(v1 & 0xf) - 8);
      const float f3 = deltaY * ((int8_t)(v1 >> 4) - 8);

      sumf += f0 * f2 + f1 * f3;
    }
  }

  *s = sumf;
}

static MTL::Function *addFunc = nullptr;
static MTL::Device *device = nullptr;
static MTL::ComputePipelineState *state = nullptr;
static MTL::CommandQueue *commandQueue = nullptr;
static MTL::Buffer *bufferA = nullptr;
static MTL::Buffer *bufferB = nullptr;

static void test_ggml_vec_dot_q4_0(const int n, float *s, const void *vx,
                                   const void *vy) {
  const int nb = n / QK;

  auto bufferA = device->newBuffer(vx, sizeof(block_q4_0) * nb,
                                   MTL::ResourceStorageModeShared);
  auto bufferB = device->newBuffer(vy, sizeof(block_q4_0) * nb,
                                   MTL::ResourceStorageModeShared);

  auto bufferResult =
      device->newBuffer(sizeof(float), MTL::ResourceStorageModePrivate);

  // Send a command to the GPU to perform the calculation.
  // Create a command buffer to hold commands.
  auto commandBuffer = commandQueue->commandBuffer();

  // Start a compute pass.
  auto commandEncoder = commandBuffer->computeCommandEncoder();

  // Encode the pipeline state object and its parameters.
  commandEncoder->setComputePipelineState(state);

  commandEncoder->setBuffer(bufferA, 0, 0);
  commandEncoder->setBuffer(bufferB, 0, 1);
  commandEncoder->setBuffer(bufferResult, 0, 2);

  auto gridSize = MTL::Size(nb, 1, 1);
  auto threadGroupSize = state->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > nb) {
    threadGroupSize = nb;
  }

  commandEncoder->dispatchThreads(gridSize, MTL::Size(threadGroupSize, 1, 1));
  commandEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  memcpy(s, bufferResult->contents(), sizeof(float));
}

#define TEST_SIZE 5000
int main(int argc, const char *argv[]) {
  int runs = argv[1] ? atoi(argv[1]) : 10000;

  // Generate test data
  srand(time(NULL));

  NS::Error *error = nullptr;
  auto *autoreleasePool = NS::AutoreleasePool::alloc()->init();
  auto deviceArray = MTL::CopyAllDevices();
  device = deviceArray->object<MTL::Device>(0);

  auto library = device->newLibrary(
      NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &error);
  if (!library) {
    printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }

  addFunc = library->newFunction(
      NS::String::string("add_arrays", NS::UTF8StringEncoding));

  state = device->newComputePipelineState(addFunc, &error);
  if (!state) {
    printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }

  commandQueue = device->newCommandQueue();

  bufferA = device->newBuffer(sizeof(block_q4_0) * 5120,
                              MTL::ResourceStorageModeManaged);
  bufferB = device->newBuffer(sizeof(block_q4_0) * 5120,
                              MTL::ResourceStorageModeManaged);

  block_q4_0 x[TEST_SIZE];
  block_q4_0 y[TEST_SIZE];
  for (int i = 0; i < TEST_SIZE; i++) {
    x[i].d = 0.1;
    y[i].d = 0.2;
    for (int j = 0; j < QK / 2; j++) {
      x[i].qs[j] = 1 & 0xff;
      y[i].qs[j] = 2 & 0xff;
    }
  }

  for (int i = 0; i < runs; i++) {
    float s = 0;
    if (0) {
      ggml_vec_dot_q4_0(64, &s, x, y);
    } else {
      test_ggml_vec_dot_q4_0(64, &s, x, y);
    }

    if (i % 100 == 0) {
      float percent = (float)i / (float)runs * 100.0;
      printf("\r%3.0f%%", percent);
      fflush(stdout);
    }
  }

  addFunc->release();
  library->release();
  autoreleasePool->release();

  return 0;
}
