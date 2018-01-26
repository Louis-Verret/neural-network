#include "GPU.h"

#include "../Common/device_picker.hpp"

extern int DEVICE;

namespace GPU
{
        void init() {
            initContext();
            initProgram();
            initQueue();
            initMatMulKernel();
        }

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
            static cl::Program program(GPU::context, util::loadProgram("../src/matmul.cl"), true);
            return program;
        }

        const cl::CommandQueue initQueue() {
            std::vector<cl::Device> devices;
            getDeviceList(devices);
            cl::Device device = devices[DEVICE];
            cl::CommandQueue queue(GPU::context, device);
            return queue;
        }

        const cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>  initMatMulKernel() {
            static cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul(GPU::program, "mmul");
            return mmul;
        }

        cl::Context context = initContext();
        cl::Program program = initProgram();
        cl::CommandQueue queue = initQueue();
        cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mat_mul_kernel = initMatMulKernel();
}
