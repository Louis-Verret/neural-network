#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "Matrix.h"
#include "ActivationFunction.h"

class Layer
{
public:
    Layer(int input_dim, int neurons_number, char const* function_name);
    ~Layer();

    int getNeuronsNumber() const {return m_neurons_number;};
    int getInputDim() const {return m_input_dim;};
    Matrix getWeights() const {return m_weights;};
    Matrix getLastWeights() const {return m_V_dW;};
    Matrix getLastWeights2() const {return m_S_dW;};
    void setWeights(Matrix& weights) {m_weights = weights;};
    void setLastWeights(Matrix& V_dW) {m_V_dW = V_dW;};
    void setLastWeights2(Matrix& S_dW) {m_S_dW = S_dW;};
    std::vector<double> getBias() const {return m_bias;};
    std::vector<double> getLastBias() const {return m_V_dB;};
    std::vector<double> getLastBias2() const {return m_S_dB;};
    void setBias(std::vector<double>& bias) {m_bias = bias;};
    void setLastBias(std::vector<double>& V_dB) {m_V_dB = V_dB;};
    void setLastBias2(std::vector<double>& S_dB) {m_S_dB = S_dB;};
    void setDropout(double dropout_rate) {m_dropout_rate = dropout_rate;};
    ActivationFunction* getActivationFunction() const {return m_f;};

    Matrix multiply(const Matrix& input);
    Matrix add(Matrix v);
    Matrix activate(const Matrix& x);

protected:
    int m_input_dim;
    int m_neurons_number;
    Matrix m_weights;
    Matrix m_V_dW;
    Matrix m_S_dW;
    std::vector<double> m_bias;
    std::vector<double> m_V_dB;
    std::vector<double> m_S_dB;
    ActivationFunction* m_f;
    double m_dropout_rate = 0;
};

std::ostream& operator << (std::ostream& out, const Layer& layer);

#endif
