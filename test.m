#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>

void add_arrays(const float* inA, const float* inB, float* result, int length) {
  for (int index = 0; index < length; index++) {
    result[index] = inA[index] + inB[index];
  }
}

// The number of floats in each array, and the size of the arrays in bytes.
const unsigned int arrayLength = 16;
const unsigned int bufferSize = arrayLength * sizeof(float);

int main(void) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  NSError* error = nil;

  NSURL* url = [NSURL fileURLWithPath:@"default.metallib"];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library || error) {
    printf("Error occurred when creating default library: %s\n", [[error description] UTF8String]);
    puts("Failed to find the default library.");
    exit(EXIT_FAILURE);
  }

  id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
  if (!function) {
    puts("Failed to find the adder function.");
    exit(EXIT_FAILURE);
  }

  // Create a compute pipeline state object.
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (!pipeline) {
    //  If the Metal API validation is enabled, you can find out more
    //  information about what went wrong.  (Metal API validation is enabled
    //  by default when a debug build is run from Xcode)
    printf("Failed to created pipeline state object, error %s.", [[error description] UTF8String]);
    exit(EXIT_FAILURE);
  }

  id<MTLCommandQueue> commandQueue = [device newCommandQueue];
  assert(commandQueue);
  if (!commandQueue) {
    puts("Failed to find the command queue.");
    exit(EXIT_FAILURE);
  }

  // Create buffers to hold data
  // Allocate three buffers to hold our initial data and the result.
  id<MTLBuffer> bufferA = [device newBufferWithLength:bufferSize
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferB = [device newBufferWithLength:bufferSize
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferResult = [device newBufferWithLength:bufferSize
                                                   options:MTLResourceStorageModeShared];

  for (unsigned long index = 0; index < arrayLength; index++) {
    float* dataPtr = bufferA.contents;
    dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    dataPtr = bufferB.contents;
    dataPtr[index] = (float)rand() / (float)(RAND_MAX);
  }

  // Send a command to the GPU to perform the calculation.
  // Create a command buffer to hold commands.
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  assert(commandBuffer);

  // Start a compute pass.
  id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
  assert(computeEncoder);

  // Encode the pipeline state object and its parameters.
  [computeEncoder setComputePipelineState:pipeline];
  [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
  [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
  [computeEncoder setBuffer:bufferResult offset:0 atIndex:2];

  // Encode the compute command.
  [computeEncoder dispatchThreads:MTLSizeMake(arrayLength, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

  // End the compute pass.
  [computeEncoder endEncoding];

  // Execute the command.
  [commandBuffer commit];

  // Normally, you want to do other work in your app while the GPU is running,
  // but in this example, the code simply blocks until the calculation is
  // complete.
  [commandBuffer waitUntilCompleted];

  float* a = bufferA.contents;
  float* b = bufferB.contents;
  float* result = bufferResult.contents;

  for (unsigned long index = 0; index < arrayLength; index++) {
    if (result[index] != (a[index] + b[index])) {
      printf("Compute ERROR: index=%lu result=%g vs %g=a+b\n", index, result[index],
             a[index] + b[index]);
      assert(result[index] == (a[index] + b[index]));
    }
  }
  printf("Compute results as expected\n");

  return 0;
}
