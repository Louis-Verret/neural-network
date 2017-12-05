#include "NeuralNetwork.h"
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NeuralNetwork::NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<double>& d, int s) {
    srand(time(NULL));
    generateData(x, d, s);
    int n = 1;
    for (int i = 0; i<n; i++) {
        m_weights.push_back(((double)rand() / (double)RAND_MAX) * 1.0);
        //std::cout << m_weights[i] << std::endl;
    }
}

void NeuralNetwork::generateData(std::vector<std::vector<double> >& x, std::vector<double>& d, int s) {
    for (int i = 0; i<s; i++) {
        std::vector<double> xi;
        double input = ((double)rand() / (double)RAND_MAX * 20) - 10;
        xi.push_back(input);
        d.push_back(3 * input + 4);
        x.push_back(xi);
    }
}

void NeuralNetwork::fit(std::vector<std::vector<double> >& x, std::vector<double>& d, int epoch, const double learning_rate) {
    int s = x.size();
    for (int i = 0; i<epoch; i++) {
        double error = 0;
        double y;
        for (int j = 0; j<s; j++) {
            y = propagate(x[j]);
            backpropagate(x[j], y, d[j], learning_rate);
            error += (d[j] - y) * (d[j] - y);
        }
        std::cout << "Error: " << error << std::endl;
    }

}

double NeuralNetwork::predict(std::vector<double>& xi) {
    return propagate(xi);
}

double NeuralNetwork::propagate(const std::vector<double>& xi) const {
    double y = 0;
    int n = xi.size();
    for (int j = 0; j < n; j++) {
        y += m_weights[j] * xi[j];
    }
    //return 1.0 / (1.0 + exp(-y));
    return y;
}

void NeuralNetwork::backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate) {

    int n = m_weights.size();
    for (int i = 0; i<n; i++) {
        m_weights[i] += learning_rate * (di - y) * xi[i];
    }

}
