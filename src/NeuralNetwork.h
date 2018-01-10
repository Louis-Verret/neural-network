#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

#include "Layer.h"
#include "Optimizer.h"
#include "CostFunction.h"
#include "Metric.h"

class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(Optimizer* optimizer, char const* cost_name, char const* metric_name = "none");
    void fit(Matrix& x, Matrix& y, int epoch, const int batch_size);
    Matrix predict(Matrix& xi);
    void addLayer(int neurons_number, char const* function_name, int input_dim = 0);
    void addDropout(double dropout_rate);
    void save(const char* file_name);
    void load(const char* file_name);

    std::vector<Layer*> getLayers() const {return m_layers;};

protected:
    std::vector<Layer*> m_layers;
    std::vector<Matrix> m_z;
    std::vector<Matrix> m_a;
    Optimizer* m_optimizer;
    CostFunction* m_C;
    Metric* m_metric = NULL;
    int input_dim;

    void propagate(Matrix& input);
    void backpropagate(const Matrix& y, const int batch_size, int epoch_num);
    void separateDataInBatches(Matrix& x, Matrix& y, std::vector<Matrix>& batches_x, std::vector<Matrix>& batches_y, const int batch_size);
};

std::ostream& operator << (std::ostream& out, const NeuralNetwork& net);

#endif
