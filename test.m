#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#define CHECK_ERROR(ptr)                                                                         \
  if (!(ptr) || error) {                                                                         \
    fprintf(stderr, "%s:%d: error: %s\n", __FILE__, __LINE__, [[error description] UTF8String]); \
    exit(1);                                                                                     \
  }

int main(int argc, const char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  NSError *error = nil;

  NSURL *url = [NSURL fileURLWithPath:@"default.metallib"];
  printf("loading library from %s\n", [[url absoluteString] UTF8String]);
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  CHECK_ERROR(library);

  id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
  CHECK_ERROR(function);

  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  CHECK_ERROR(pipeline)

  id<MTLCommandQueue> queue = [device newCommandQueue];

  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

  float x_data[] = {1, 2, 3, 4};
  float y_data[] = {5, 6, 7, 8};
  float z_data[] = {0, 0, 0, 0};

  size_t len = 4;
  id<MTLBuffer> x = [device newBufferWithBytes:x_data
                                        length:sizeof(x_data)
                                       options:MTLResourceStorageModeShared];
  id<MTLBuffer> y = [device newBufferWithBytes:y_data
                                        length:sizeof(y_data)
                                       options:MTLResourceStorageModeShared];
  id<MTLBuffer> z = [device newBufferWithBytes:z_data
                                        length:sizeof(z_data)
                                       options:MTLResourceStorageModeShared];

  {
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:y offset:0 atIndex:1];
    [encoder setBuffer:z offset:0 atIndex:2];
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(len, 1, 1)];
  };

  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  MTLCommandBufferStatus status = (MTLCommandBufferStatus)[command_buffer status];
  if (status != MTLCommandBufferStatusCompleted) {
    fprintf(stderr, "%s: command buffer failed with status %lu\n", __func__, status);
    exit(1);
  }

  for (size_t i = 0; i < len; i++) {
    uint8_t *ptr_z = z.contents;
    printf("%d\n", ptr_z[i]);
  }

  printf("done\n");
  return 0;
}
