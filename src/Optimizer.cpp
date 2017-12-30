#include "Optimizer.h"
#include "Utils.h"

Optimizer::Optimizer()
{

}

Optimizer::~Optimizer()
{

}

SGD::SGD(const double learning_rate, const double momentum, const double decay) :
 m_learning_rate(learning_rate),
 m_momentum(momentum),
 m_decay(decay)
{

}

SGD::~SGD()
{

}

void SGD::updateWeights(Layer* layer, const Matrix& a, const Matrix& delta, int batch_size, int epoch_num) const {
    Matrix dW = (delta * a.transpose()) / batch_size;
    Matrix V_dW = dW + m_momentum * (*layer->getLastWeights());
    Matrix new_weights = (*layer->getWeights()) - (1 / (1 + m_decay * epoch_num)) * V_dW;
    layer->setLastWeights(V_dW);
    layer->setWeights(new_weights);
}

void SGD::updateBias(Layer* layer, const Matrix& delta, int batch_size, int epoch_num) const {
    std::vector<double> ones(delta.getM(), 1);
    std::vector<double> dB = (delta * ones) / batch_size;
    std::vector<double> V_dB = dB + m_momentum * layer->getLastBias();
    std::vector<double> new_bias = layer->getBias() - m_learning_rate * V_dB;
    layer->setLastBias(V_dB);
    layer->setBias(new_bias);
}

Adam::Adam(const double learning_rate, const double beta_1, const double beta_2, const double epsilon, const double decay) :
 m_learning_rate(learning_rate),
 m_beta_1(beta_1),
 m_beta_2(beta_2),
 m_epsilon(epsilon),
 m_decay(decay)
{

}

Adam::~Adam()
{

}

void Adam::updateWeights(Layer* layer, const Matrix& a, const Matrix& delta, int batch_size, int epoch_num) const {
    Matrix dW = (delta * a.transpose()) / batch_size;
    Matrix V_dW = (1 - m_beta_1) * dW + m_beta_1 * (*layer->getLastWeights());
    Matrix S_dW = (1 - m_beta_2) * dW.hadamardProduct(dW) + m_beta_2 * (*layer->getLastWeights2());
    Matrix V_dW_corrected = V_dW / (1 - std::pow(m_beta_1, epoch_num));
    Matrix S_dW_corrected = S_dW / (1 - std::pow(m_beta_2, epoch_num));
    Matrix new_weights = (*layer->getWeights()) - (1 / (1 + m_decay * epoch_num)) * m_learning_rate * (V_dW_corrected / (S_dW_corrected.sqrt() + m_epsilon));
    layer->setLastWeights(V_dW);
    layer->setLastWeights2(S_dW);
    layer->setWeights(new_weights);
}

void Adam::updateBias(Layer* layer, const Matrix& delta, int batch_size, int epoch_num) const {
    std::vector<double> ones(delta.getM(), 1);
    std::vector<double> dB = (delta * ones) / batch_size;
    std::vector<double> V_dB = (1 - m_beta_1) * dB + m_beta_1 * layer->getLastBias();
    std::vector<double> S_dB = (1 - m_beta_2) * (dB*dB) + m_beta_2 * layer->getLastBias2();
    std::vector<double> V_dB_corrected = V_dB / (1 - std::pow(m_beta_1, epoch_num));
    std::vector<double> S_dB_corrected = S_dB / (1 - std::pow(m_beta_2, epoch_num));
    std::vector<double> new_bias = layer->getBias() - m_learning_rate * (V_dB_corrected / (sqrt(S_dB_corrected) + m_epsilon));
    layer->setLastBias(V_dB);
    layer->setLastBias2(S_dB);
    layer->setBias(new_bias);
}
