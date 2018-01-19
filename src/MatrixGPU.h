
#ifndef MATRIXGPU
#define MATRIXGPU

#include <vector>

int DEVICE = 0;

class MatrixGPU {
public:
    MatrixGPU();
    MatrixGPU(cl::Context& context, int n, int m);
    // MatrixGPU(const MatrixGPU& mat);
    ~MatrixGPU();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    cl::Buffer getBuff() const { return m_buff; };
    const double &operator()(cl::CommandQueue& queue, int i, int j) const;
    MatrixGPU matmult(cl::CommandQueue& queue, cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> kernel,
                      const MatrixGPU &mat) const;

protected:
    int m_n;
    int m_m;
    cl::Context m_context;
    cl::Buffer m_buff;
};


#endif //MATRIXGPU
