#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "Matrix.h"
#include "Vector.h"
#include "ActivationFunction.h"


/** Class that implements a Layer.
    Each layer basically contains a weight matrix and a bias vector
    which are updated during the backpropagation
**/
class Layer
{
public:

    /* Constructor / Destructor */
    Layer(int input_dim, int neurons_number, char const* function_name, bool init = true);
    ~Layer();

    /* Get Methods */
    int getNeuronsNumber() const {return m_neurons_number;};
    int getInputDim() const {return m_input_dim;};
    Matrix getWeights() const {return m_weights;};
    Matrix getLastWeights() const {return m_V_dW;};
    Matrix getLastWeights2() const {return m_S_dW;};
    Vector getBias() const {return m_bias;};
    Vector getLastBias() const {return m_V_dB;};
    Vector getLastBias2() const {return m_S_dB;};
    ActivationFunction* getActivationFunction() const {return m_f;};

    /* Set Methods */
    void setWeights(Matrix& weights) {m_weights = weights;};
    void setLastWeights(Matrix& V_dW) {m_V_dW = V_dW;};
    void setLastWeights2(Matrix& S_dW) {m_S_dW = S_dW;};
    void setBias(Vector& bias) {m_bias = bias;};
    void setLastBias(Vector& V_dB) {m_V_dB = V_dB;};
    void setLastBias2(Vector& S_dB) {m_S_dB = S_dB;};
    void setDropout(double dropout_rate) {m_dropout_rate = dropout_rate;};

    /* Methods used during the propagation step */
    Matrix multiply(const Matrix& input);
    Matrix add(Matrix v);
    Matrix activate(const Matrix& x);

protected:
    int m_input_dim; // Dimension of the previous layer
    int m_neurons_number; // Dimension of the current layer

    Matrix m_weights; // Weight matrix of size m_neurons_number * m_input_dim
    Vector m_bias; // Bias vector of size m_neurons_number
    Matrix m_V_dW; // Weight matrix at the previous iteration (used by Adam optimizer)
    Matrix m_S_dW; // Squared weight matrix at the previous iteration (used by Adam optimizer)
    Vector m_V_dB; // Bias vector at the previous iteration (used by Adam optimizer)
    Vector m_S_dB; // Squared bias vector at the previous iteration (used by Adam optimizer)

    ActivationFunction* m_f; // Activation function for this layer
    
    double m_dropout_rate = 0; // Dropout rate (between 0 and 1)
};

std::ostream& operator << (std::ostream& out, const Layer& layer);

#endif
