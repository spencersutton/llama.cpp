/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A shader that adds two arrays of floats.
*/

#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
kernel void add_arrays(device const uchar* inA,
                       device const uchar* inB,
                       device uchar* result,
                       uint index [[thread_position_in_threadgroup]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    result[index] =(float)index;
}
