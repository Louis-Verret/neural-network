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

        const cl::make_kernel<int, int, cl::Buffer, cl::Buffer>  initMatSumElemKernel() {
            static cl::make_kernel<int, int, cl::Buffer, cl::Buffer> sumElem(GPU::program, "sum_elements");
            return sumElem;
        }

        cl::Context context = initContext();
        cl::Program program = initProgram();
        cl::CommandQueue queue = initQueue();
        cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mmul_kernel = initMatMMulKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_tranpose_kernel = initMatTransposeKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_add_kernel = initMatAddKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_sub_kernel = initMatSubKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mul_kernel = initMatMulKernel();
        cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mat_sum_elem_kernel = initMatSumElemKernel();

}
