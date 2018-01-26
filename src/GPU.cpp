#include "GPU.h"

#include "../Common/device_picker.hpp"

extern int DEVICE;

namespace GPU
{
        const cl::Context initContext() {
            std::vector<cl::Device> devices;
            getDeviceList(devices);
            cl::Device device = devices[DEVICE];
            std::vector<cl::Device> chosen_device;
            chosen_device.push_back(device);
            static cl::Context context(chosen_device);
            return context;
        }

        const cl::Program initProgram() {
            static cl::Program program(GPU::context, util::loadProgram("../src/kernels.cl"), true);
            return program;
        }

        const cl::CommandQueue initQueue() {
            std::vector<cl::Device> devices;
            getDeviceList(devices);
            cl::Device device = devices[DEVICE];
            cl::CommandQueue queue(GPU::context, device);
            return queue;
        }

        const cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatMMulKernel() {
            static cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul(GPU::program, "mmul");
            return mmul;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatVMulKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> vmul(GPU::program, "vmul");
            return vmul;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatAddVecKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> add_vector(GPU::program, "add_vector");
            return add_vector;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatTransposeKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> transpose(GPU::program, "transpose_naive");
            return transpose;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> initMatAddKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> add(GPU::program, "add");
            return add;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> initMatSubKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> sub(GPU::program, "sub");
            return sub;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> initMatMulKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mul(GPU::program, "mul");
            return mul;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> initMatDivKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> div_k(GPU::program, "div");
            return div_k;
        }

        const cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> initMatDivCoeffKernel() {
            static cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> divCoeff(GPU::program, "div_coeff");
            return divCoeff;
        }

        const cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> initMatAddCoeffKernel() {
            static cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> addCoeff(GPU::program, "add_coeff");
            return addCoeff;
        }

        const cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> initMatCoeffMulCoeffKernel() {
            static cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> coeffMul(GPU::program, "coeff_mul");
            return coeffMul;
        }

        const cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> initMatCoeffSubCoeffKernel() {
            static cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> coeffSub(GPU::program, "coeff_sub");
            return coeffSub;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer>  initMatSumElemKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer> sumElem(GPU::program, "sum_elements");
            return sumElem;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer>  initMatSqrtKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer> sqrt_k(GPU::program, "sqrt_k");
            return sqrt_k;
        }

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer>  initMatLogKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer> log_k(GPU::program, "log_k");
            return log_k;
        }

        const cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> initVecAddKernel() {
            static cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vectorAdd(GPU::program, "vector_add");
            return vectorAdd;
        }

        const cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> initVecSubKernel() {
            static cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vectorSub(GPU::program, "vector_sub");
            return vectorSub;
        }

        const cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> initVecMulKernel() {
            static cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vectorMul(GPU::program, "vector_mul");
            return vectorMul;
        }

        const cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> initVecDivKernel() {
            static cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vectorDiv(GPU::program, "vector_div");
            return vectorDiv;
        }

        const cl::make_kernel<int, cl::Buffer, double, cl::Buffer>  initVecAddCoeffKernel() {
            static cl::make_kernel<int, cl::Buffer, double, cl::Buffer> vectorAddCoeff(GPU::program, "vector_add_coeff");
            return vectorAddCoeff;
        }

        const cl::make_kernel<int, cl::Buffer, double, cl::Buffer>  initVecDivCoeffKernel() {
            static cl::make_kernel<int, cl::Buffer, double, cl::Buffer> vectorDivCoeff(GPU::program, "vector_div_coeff");
            return vectorDivCoeff;
        }

        const cl::make_kernel<int, double, cl::Buffer, cl::Buffer>  initVecCoeffMulKernel() {
            static cl::make_kernel<int, double, cl::Buffer, cl::Buffer> vectorCoeffMul(GPU::program, "vector_coeff_mul");
            return vectorCoeffMul;
        }

        const cl::make_kernel<int, cl::Buffer, cl::Buffer> initVecSqrtKernel() {
            static cl::make_kernel<int, cl::Buffer, cl::Buffer>  vectorSqrt(GPU::program, "vector_sqrt");
            return vectorSqrt;
        }

        const cl::make_kernel<int, cl::Buffer> initVecFillWithZerosKernel() {
            static cl::make_kernel<int, cl::Buffer>  vectorFillWithZeros(GPU::program, "vector_fill_with_zeros");
            return vectorFillWithZeros;
        }

        const cl::make_kernel<int, cl::Buffer, double> initVecFillWithKernel() {
            static cl::make_kernel<int, cl::Buffer, double> vectorFillWith(GPU::program, "vector_fill_with");
            return vectorFillWith;
        }


        cl::Context context = initContext();
        cl::Program program = initProgram();
        cl::CommandQueue queue = initQueue();
        cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mmul_kernel = initMatMMulKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_vmul_kernel = initMatVMulKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_add_vec_kernel = initMatAddVecKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_tranpose_kernel = initMatTransposeKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_add_kernel = initMatAddKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_sub_kernel = initMatSubKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mul_kernel = initMatMulKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_div_kernel = initMatDivKernel();
        cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> mat_div_coeff_kernel = initMatDivCoeffKernel();
        cl::make_kernel<int, int, cl::Buffer, double, cl::Buffer> mat_add_coeff_kernel = initMatAddCoeffKernel();
        cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> mat_coeff_sub_kernel = initMatCoeffSubCoeffKernel();
        cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer> mat_coeff_mul_kernel = initMatCoeffMulCoeffKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_sum_elem_kernel = initMatSumElemKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_sqrt_kernel = initMatSqrtKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_log_kernel = initMatLogKernel();

        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_add_kernel = initVecAddKernel();
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_sub_kernel = initVecSubKernel();
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_mul_kernel = initVecMulKernel();
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> vec_div_kernel = initVecDivKernel();
        cl::make_kernel<int, cl::Buffer, double, cl::Buffer> vec_add_coeff_kernel = initVecAddCoeffKernel();
        cl::make_kernel<int, cl::Buffer, double, cl::Buffer> vec_div_coeff_kernel = initVecDivCoeffKernel();
        cl::make_kernel<int, double, cl::Buffer, cl::Buffer> vec_coeff_mul_kernel = initVecCoeffMulKernel();
        cl::make_kernel<int, cl::Buffer, cl::Buffer> vec_sqrt_kernel = initVecSqrtKernel();
        cl::make_kernel<int, cl::Buffer> vec_fill_with_zeros_kernel = initVecFillWithZerosKernel();
        cl::make_kernel<int, cl::Buffer, double> vec_fill_with_kernel = initVecFillWithKernel();

}
