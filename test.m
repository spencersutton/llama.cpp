#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main(int argc, const char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  NSError *error = nil;
  NSURL *url = [NSURL fileURLWithPath:@"test.metal"];

  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (error) {
    fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
    exit(1);
  }

  id<MTLFunction> function = [library newFunctionWithName:@"kernel_function"];
  if (!function) {
    fprintf(stderr, "%s: error: %s\n", __func__, "no kernel function");
    exit(1);
  }

  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (error) {
    fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
    exit(1);
  }

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
