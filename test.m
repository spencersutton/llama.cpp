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
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:nil];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  return 0;
}
