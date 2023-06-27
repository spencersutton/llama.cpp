#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main(int argc, const char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  NSError *error = nil;

  NSURL *url = [NSURL fileURLWithPath:@"test.metal"];
  fprintf(stderr, "%s: loading '%s'\n", __func__, [url absoluteString].UTF8String);

  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (error) {
    fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
    exit(1);
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];

  id<MTLFunction> function = [library newFunctionWithName:@"kernel_function"];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:nil];

  return 0;
}
