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

  {};

  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  printf("done\n");
  return 0;
}
