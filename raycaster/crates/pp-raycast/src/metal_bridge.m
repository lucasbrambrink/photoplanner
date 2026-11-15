#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

@interface PpMetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLLibrary> library;
@end

@implementation PpMetalContext
@end

static void copy_cstr(char* dst, size_t cap, const char* src) {
    if (!dst || cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
}

void* pp_metal_create(
    const char* msl_source,
    char* device_name_out,
    size_t name_cap,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            copy_cstr(err_out, err_cap, "MTLCreateSystemDefaultDevice returned nil");
            return NULL;
        }
        copy_cstr(device_name_out, name_cap, [[device name] UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            copy_cstr(err_out, err_cap, "Failed to create MTLCommandQueue");
            return NULL;
        }

        NSString* src = [NSString stringWithUTF8String:msl_source];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;

        NSError* err = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:src options:opts error:&err];
        if (!library) {
            const char* msg = err ? [[err localizedDescription] UTF8String] : "Unknown Metal compile error";
            copy_cstr(err_out, err_cap, msg);
            return NULL;
        }

        PpMetalContext* ctx = [[PpMetalContext alloc] init];
        ctx.device = device;
        ctx.queue = queue;
        ctx.library = library;
        return (__bridge_retained void*)ctx;
    }
}

void pp_metal_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    @autoreleasepool {
        PpMetalContext* _drop = (__bridge_transfer PpMetalContext*)ctx_ptr;
        (void)_drop;
    }
}

void* pp_metal_create_pipeline(
    void* ctx_ptr,
    const char* fn_name,
    char* err_out,
    size_t err_cap
) {
    if (!ctx_ptr || !fn_name) return NULL;
    @autoreleasepool {
        PpMetalContext* ctx = (__bridge PpMetalContext*)ctx_ptr;
        NSString* name = [NSString stringWithUTF8String:fn_name];
        id<MTLFunction> fn = [ctx.library newFunctionWithName:name];
        if (!fn) {
            copy_cstr(err_out, err_cap, "Kernel function not found in MTLLibrary");
            return NULL;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            const char* msg = err ? [[err localizedDescription] UTF8String] : "Failed to create pipeline state";
            copy_cstr(err_out, err_cap, msg);
            return NULL;
        }
        return (__bridge_retained void*)pipe;
    }
}

void pp_metal_destroy_pipeline(void* pipe_ptr) {
    if (!pipe_ptr) return;
    @autoreleasepool {
        id<MTLComputePipelineState> _drop = (__bridge_transfer id<MTLComputePipelineState>)pipe_ptr;
        (void)_drop;
    }
}

void* pp_metal_create_buffer(void* ctx_ptr, const void* initial_bytes, size_t byte_len) {
    if (!ctx_ptr) return NULL;
    @autoreleasepool {
        PpMetalContext* ctx = (__bridge PpMetalContext*)ctx_ptr;
        size_t alloc_len = byte_len > 0 ? byte_len : 16;
        id<MTLBuffer> buf = nil;
        if (initial_bytes != NULL && byte_len > 0) {
            buf = [ctx.device newBufferWithBytes:initial_bytes
                                          length:alloc_len
                                         options:MTLResourceStorageModeShared];
        } else {
            buf = [ctx.device newBufferWithLength:alloc_len
                                          options:MTLResourceStorageModeShared];
        }
        if (!buf) return NULL;
        return (__bridge_retained void*)buf;
    }
}

void* pp_metal_buffer_contents(void* buf_ptr) {
    if (!buf_ptr) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_ptr;
    return [buf contents];
}

void pp_metal_destroy_buffer(void* buf_ptr) {
    if (!buf_ptr) return;
    @autoreleasepool {
        id<MTLBuffer> _drop = (__bridge_transfer id<MTLBuffer>)buf_ptr;
        (void)_drop;
    }
}

int pp_metal_dispatch_1d(
    void* ctx_ptr,
    void* pipe_ptr,
    void* const* buffers,
    size_t num_buffers,
    size_t num_threads
) {
    if (!ctx_ptr || !pipe_ptr) return -1;
    if (num_threads == 0) return 0;
    @autoreleasepool {
        PpMetalContext* ctx = (__bridge PpMetalContext*)ctx_ptr;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipe_ptr;

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        if (!cmd) return -2;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -3;

        [enc setComputePipelineState:pipe];
        for (size_t i = 0; i < num_buffers; i++) {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffers[i];
            [enc setBuffer:b offset:0 atIndex:i];
        }

        NSUInteger maxTg = [pipe maxTotalThreadsPerThreadgroup];
        NSUInteger tgWidth = maxTg < 256 ? maxTg : 256;
        if (tgWidth == 0) tgWidth = 64;
        NSUInteger numGroups = (num_threads + tgWidth - 1) / tgWidth;

        [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgWidth, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -4;
    }
}

int pp_metal_dispatch_2pass_2d(
    void* ctx_ptr,
    void* pipe1_ptr,
    void* const* bufs1,
    size_t nbufs1,
    void* pipe2_ptr,
    void* const* bufs2,
    size_t nbufs2,
    size_t width,
    size_t height
) {
    if (!ctx_ptr || !pipe1_ptr) return -1;
    if (width == 0 || height == 0) return 0;
    @autoreleasepool {
        PpMetalContext* ctx = (__bridge PpMetalContext*)ctx_ptr;
        id<MTLComputePipelineState> pipe1 = (__bridge id<MTLComputePipelineState>)pipe1_ptr;

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        if (!cmd) return -2;

        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!enc) return -3;
            [enc setComputePipelineState:pipe1];
            for (size_t i = 0; i < nbufs1; i++) {
                id<MTLBuffer> b = (__bridge id<MTLBuffer>)bufs1[i];
                [enc setBuffer:b offset:0 atIndex:i];
            }
            NSUInteger w = [pipe1 threadExecutionWidth];
            if (w == 0) w = 16;
            NSUInteger h = [pipe1 maxTotalThreadsPerThreadgroup] / w;
            if (h > 8) h = 8;
            if (h == 0) h = 1;
            MTLSize tg = MTLSizeMake(w, h, 1);
            MTLSize groups = MTLSizeMake((width + w - 1) / w, (height + h - 1) / h, 1);
            [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        if (pipe2_ptr != NULL) {
            id<MTLComputePipelineState> pipe2 = (__bridge id<MTLComputePipelineState>)pipe2_ptr;
            id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
            if (!enc2) return -3;
            [enc2 setComputePipelineState:pipe2];
            for (size_t i = 0; i < nbufs2; i++) {
                id<MTLBuffer> b = (__bridge id<MTLBuffer>)bufs2[i];
                [enc2 setBuffer:b offset:0 atIndex:i];
            }
            NSUInteger w = [pipe2 threadExecutionWidth];
            if (w == 0) w = 16;
            NSUInteger h = [pipe2 maxTotalThreadsPerThreadgroup] / w;
            if (h > 8) h = 8;
            if (h == 0) h = 1;
            MTLSize tg = MTLSizeMake(w, h, 1);
            MTLSize groups = MTLSizeMake((width + w - 1) / w, (height + h - 1) / h, 1);
            [enc2 dispatchThreadgroups:groups threadsPerThreadgroup:tg];
            [enc2 endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -4;
    }
}
