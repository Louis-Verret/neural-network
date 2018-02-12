#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

#include "Layer.h"
#include "Optimizer.h"
#include "CostFunction.h"
#include "Metric.h"
#include "MatrixCPU.h"

/** Class that implements a neural network
    This is the central class and allows the user to construct, train
    and validate a neural network through public methods
    The user can specify the activation function in each layer,
    the error function, the metric and the optimizer for training
**/
class NeuralNetwork
{
public:

    /* Constructors / Destructor */
    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(Optimizer* optimizer, char const* cost_name, char const* metric_name = "none");

    /* Methods for constructing and training a neural network */
    void fit(MatrixCPU& x, MatrixCPU& y, int epoch, const int batch_size);
    void validate(Matrix& x, Matrix& y);
    Matrix predict(Matrix& xi);
    void addLayer(int neurons_number, char const* function_name, int input_dim = 0);

    /* Get method */
    std::vector<Layer*> getLayers() const {return m_layers;};

protected:

    std::vector<Layer*> m_layers; // List of Layer
    std::vector<Matrix> m_z; // List of results at each layer before activation
    std::vector<Matrix> m_a; // List of results at each layer after activation

    Optimizer* m_optimizer; // chosen optimizer
    CostFunction* m_C; // chosen error function
    Metric* m_metric = NULL; // chosen metric
    int input_dim; // input of the data

    void propagate(Matrix& input); // propagation algorithm
    void backpropagate(const Matrix& y, const int batch_size, int epoch_num); // backpropagation algorithm

    void separateDataInBatches(MatrixCPU& x, MatrixCPU& y, std::vector<Matrix>& batches_x, std::vector<Matrix>& batches_y, const int batch_size);
};

std::ostream& operator << (std::ostream& out, const NeuralNetwork& net);

#endif
