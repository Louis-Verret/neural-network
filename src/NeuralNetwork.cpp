#include "NeuralNetwork.h"
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NeuralNetwork::NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<double>& d, int s) : m_hidden_layers(-1) {
    srand(time(NULL));
    generateData(x, d, s);
    addLayer(1);
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

void NeuralNetwork::addLayer(int size) {
    m_hidden_layers++;
    std::vector<double> layer;
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        layer.push_back(((double)rand() / (double)RAND_MAX));
    }
    m_layers.push_back(layer);
}

double NeuralNetwork::propagate(const std::vector<double>& xi) const {
    int n = xi.size();
    int n_layer = m_layers[1].size();
    std::vector<double> y;
    for (int j = 0; j < n_layer; j++) {
        double y1j = 0;
        for (int k = 0; k < n; k++) {
            y1j += m_layers[0][k] * xi[k];
        }
        y.push_back(y1j);
    }
    double ys = 0;
    for (int k = 0; k < n_layer; k++) {
        ys += m_layers[1][k] * y[k];
    }
    return ys;
}

void NeuralNetwork::backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate) {

    int n = m_layers[0].size();
    for (int i = 0; i<n; i++) {
        m_layers[0][i] += learning_rate * (di - y) * xi[i];
    }

}
