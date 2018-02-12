#ifndef GPU_H
#define GPU_H

#include "../Common/cl.hpp"
#include "../Common/util.hpp"

/** Namespace useful for accessing already compiled kernels and
    OpenCL methods and fields as global variables **/

namespace GPU
{

    /* OpenCL fields */
    extern cl::Context context;
    extern cl::Program program;
    extern cl::CommandQueue queue;

    /* Matrix Kernels */
    extern cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>  mat_mmul_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer>  mat_vmul_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_add_vec_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_tranpose_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_sum_elem_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_add_kernel;
    extern cl::make_kernel<int, int, cl::Buffer> mat_fill_with_zeros_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_sub_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mul_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_div_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, double, cl::Buffer> mat_div_coeff_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, double, cl::Buffer> mat_add_coeff_kernel;
    extern cl::make_kernel<int, int, int, int, double, cl::Buffer, cl::Buffer> mat_coeff_sub_kernel;
    extern cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> mat_coeff_mul_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_sqrt_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_log_kernel;

    /* Vector Kernels */
    extern cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_add_kernel;
    extern cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_sub_kernel;
    extern cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_mul_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> vec_div_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> vec_add_coeff_kernel;
    extern cl::make_kernel<int, cl::Buffer, double, cl::Buffer> vec_div_coeff_kernel;
    extern cl::make_kernel<int, double, cl::Buffer, cl::Buffer> vec_coeff_mul_kernel;
    extern cl::make_kernel<int, cl::Buffer, cl::Buffer> vec_sqrt_kernel;
    extern cl::make_kernel<int, cl::Buffer> vec_fill_with_zeros_kernel;
    extern cl::make_kernel<int, int, cl::Buffer, double> vec_fill_with_kernel;

    /* Activation functions kernels */
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_linear_dev_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_sigmoid_eval_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_sigmoid_dev_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_tanh_eval_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_tanh_dev_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_relu_eval_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer> mat_relu_dev_kernel;
    extern cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_softmax_eval_kernel;

}

#endif //GPU_H
