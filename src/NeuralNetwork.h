#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

#include "Layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    void fit(Matrix& x, Matrix& d, int epoch, const double learning_rate, const int batch_size, double momentum);
    Matrix predict(Matrix& xi);
    void addLayer(int neurons_number, std::string function_name, int input_dim = 0);
    void save(const char* file_name);

    std::vector<Layer*> getLayers() const {return m_layers;};

protected:
    std::vector<Layer*> m_layers;
    std::vector<Matrix> m_z;
    std::vector<Matrix> m_a;
    int input_dim;

    Matrix computeGradient(const Matrix& d);
    void propagate(Matrix& input);
    void backpropagate(const Matrix& d, const double learning_rate, int batch_size, double momentum);
    void separateDataInBatches(Matrix& x, Matrix& d, std::vector<Matrix>& batches_x, std::vector<Matrix>& batches_d, const int batch_size);
};

std::ostream& operator << (std::ostream& out, const NeuralNetwork& net);

#endif
