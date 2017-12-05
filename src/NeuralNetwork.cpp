#include "NeuralNetwork.h"
#include "math.h"


double NeuralNetwork::propagate(const std::vector<double>& x) const {
    double y = 0;
    int m = x.size();
    for (int j = 0; j < m; j++) {
        y += m_weights[j] * x[j];
    }
    return 1 / (1 + exp(-y));
}


void NeuralNetwork::backpropagate(const std::vector<double>& y) const {




}
