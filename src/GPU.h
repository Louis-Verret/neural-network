#ifndef GPU_H
#define GPU_H

#include "../Common/cl.hpp"
#include "../Common/util.hpp"

namespace GPU
{
    void init();

    const cl::Context initContext();
    const cl::Program initProgram();
    const cl::CommandQueue initQueue();
    const cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatMulKernel();

    extern cl::Context context;
    extern cl::Program program;
    extern cl::CommandQueue queue;
    extern cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>  mat_mul_kernel;
}

#endif //GPU_H
