#include "ggml.h"
#include <assert.h>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#define QK 32

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

static MTL::Function *vec_dot = nullptr;
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

  auto path =
      NS::String::string("/Users/spencer/ai/repos/llama.cpp/quant.metallib",
                         NS::UTF8StringEncoding);
  auto library = device->newLibrary((path), &error);
  if (!library) {
    printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }

  auto fn_name = NS::String::string("ggml_compute_forward_mul_mat_q_f32",
                                    NS::UTF8StringEncoding);
  vec_dot = library->newFunction(fn_name);

  state = device->newComputePipelineState(vec_dot, &error);
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
    test_ggml_vec_dot_q4_0(64, &s, x, y);
    assert(s != 0);

    if (i % 100 == 0) {
      float percent = (float)i / (float)runs * 100.0;
      printf("\r%3.0f%%", percent);
      fflush(stdout);
    }
  }

  vec_dot->release();
  library->release();
  autoreleasePool->release();

  return 0;
}
