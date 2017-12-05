#include "NeuralNetwork.h"


void fit(std::vector<double>& x, double d) {

    double y = propagate(x);
    backpropagate(x, y, d);

}


std::vector<double>& NeuralNetwork::propagate(const std::vector<double>& x) const {

}


void NeuralNetwork::backpropagate(const std::vector<double>& x, const double y, const double d) {

    n = m_weights.size();
    for (int i = 0; i<n; i++) {
        m_weights[i] += (d - y) * x[i];
    }

}
