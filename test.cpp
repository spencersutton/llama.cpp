#include <assert.h>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

const char *kernelSrc = R"(
#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;

#define QK 32
#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct ggml_tensor {
  int n_dims;
  int64_t num_elements[4]; // number of elements
  size_t num_bytes[4];  // stride in bytes:
                 // nb[0] = sizeof(type)
                 // nb[1] = nb[0]   * ne[0] + padding
                 // nb[i] = nb[i-1] * ne[i-1]

  device struct ggml_tensor *x;
  device struct ggml_tensor *y;

  device void *data;
};

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;


struct ggml_compute_params {
  int ith;
  int nth;

  // work buffer for all threads
  device void *wdata;
};

kernel void ggml_compute_forward_mul_mat_q_f32(
    const device struct ggml_compute_params  *params, const device struct ggml_tensor *src0,
    const device struct ggml_tensor *src1, device struct ggml_tensor *dst) {
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

  device void *wdata = params->wdata;
  const size_t row_size = ne00 * sizeof(block_q4_0) / QK;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    device void *src0_row =
        (device void *)((device char *)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    device char *src1_col =
        ((device char *)wdata + ((0 + i02 * ne11 + i03 * ne12 * ne11) * row_size));

    device float *dst_col = (device float *)((device char *)dst->data +
                               (i01 * nb0 + 0 * nb1 + i02 * nb2 + i03 * nb3));


    for (int64_t ic = 0; ic < ne11; ++ic) {
      const int n = ne00;
      device float * s = &dst_col[ic * ne0];
      device const void * vx = src0_row;
      device const void * vy = (device void *)(src1_col + ic * row_size);

      const int nb = n / QK;

      device const block_q4_0 * x = ( device const block_q4_0 *)vx;
      device const block_q4_0 * y = ( device const block_q4_0 *)vy;

      float sumf = 0.0;

      // scalar
      for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        device const uint8_t * p0 = x[i].qs;
        device const uint8_t * p1 = y[i].qs;

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

)";

#define QK 32

typedef struct {
  float d;            // delta
  uint8_t qs[QK / 2]; // nibbles / quants
} block_q4_0;

static void ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    const struct ggml_tensor *src1, struct ggml_tensor *dst) {
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

  GGML_ASSERT(ne02 == ne12);
  GGML_ASSERT(ne03 == ne13);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  const enum ggml_type type = src0->type;
  quantize_row_q_t const quantize_row_q = quantize_fns[type].quantize_row_q;
  vec_dot_q_t const vec_dot_q = quantize_fns[type].vec_dot_q;

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == (int)GGML_TYPE_SIZE[type]);
  GGML_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ne0 == ne01);
  GGML_ASSERT(ne1 == ne11);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == GGML_TASK_INIT) {
    char *wdata = params->wdata;
    const size_t row_size = ne10 * GGML_TYPE_SIZE[type] / GGML_BLCK_SIZE[type];

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
      for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
          quantize_row_q((float *)((char *)src1->data + i13 * nb13 +
                                   i12 * nb12 + i11 * nb11),
                         (void *)wdata, ne10);
          wdata += row_size;
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by src0 rows using ggml_vec_dot_q

  // total rows in src0
  const int nr = ne01 * ne02 * ne03;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  void *wdata = params->wdata;
  const size_t row_size = ne00 * GGML_TYPE_SIZE[type] / GGML_BLCK_SIZE[type];

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
      vec_dot_q(ne00, &dst_col[ic * ne0], src0_row,
                (void *)(src1_col + ic * row_size));
    }
  }
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

  addFunc = library->newFunction(NS::String::string(
      "ggml_compute_forward_mul_mat_q_f32", NS::UTF8StringEncoding));

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
    assert(s != 0);

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
