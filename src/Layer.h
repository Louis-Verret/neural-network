#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "Matrix.h"
#include "ActivationFunction.h"

class Layer
{
public:
    Layer(int input_dim, int neurons_number, std::string function_name);
    ~Layer();

    int getNeuronsNumber() const {return m_neurons_number;};
    int getInputDim() const {return m_input_dim;};
    Matrix* getWeights() const {return m_weights;};
    std::vector<double> getBias() const {return m_bias;};
    ActivationFunction* getActivationFunction() const {return m_f;};

    Matrix multiply(const Matrix& input);
    Matrix add(Matrix v);
    Matrix activate(const Matrix& x);

    void updateWeights(const Matrix& a, const Matrix& delta, double learning_rate, int mini_batch, double momentum);
    void updateBias(const Matrix& delta, double learning_rate, int mini_batch, double momentum);

protected:
    Matrix* m_weights;
    Matrix* m_last_update_weights;
    std::vector<double> m_bias;
    std::vector<double> m_last_update_bias;
    int m_input_dim;
    int m_neurons_number;
    ActivationFunction* m_f;
};

std::ostream& operator << (std::ostream& out, const Layer& layer);

#endif
