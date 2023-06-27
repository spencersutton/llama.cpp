#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    NSError *error = nil;

    NSURL *url = [NSURL fileURLWithPath:@"default.metallib"];
    printf("loading library from %s\n", [[url absoluteString] UTF8String]);
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil || error != nil) {
      fprintf(stderr, "%s:%d: error: %s\n", __FILE__, __LINE__, [[error description] UTF8String]);
      exit(1);
    }

    id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
    if (function == nil) {
      fprintf(stderr, "%s:%d: error: function not found\n", __FILE__, __LINE__);
      exit(1);
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (pipeline == nil || error != nil) {
      fprintf(stderr, "%s:%d: error: %s\n", __FILE__, __LINE__, [[error description] UTF8String]);
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
  };
  return 0;
}
