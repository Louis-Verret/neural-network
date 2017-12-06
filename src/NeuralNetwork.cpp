#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>


NeuralNetwork::NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<double>& d, int s) {
    generateData(x, d, s);
}

void NeuralNetwork::generateData(std::vector<std::vector<double> >& x, std::vector<double>& d, int s) {
    srand(time(NULL));
    double max_input = -10;
    double min_input = 10;
    double max_d = -1;
    double min_d = 1;
    for (int i = 0; i<s; i++) {
        std::vector<double> xi;
        double input = ((double)rand() / (double)RAND_MAX) * 6.28 - 3.14;
        max_input = std::max(max_input, input);
        min_input = std::min(min_input, input);
        max_d = std::max(max_d, sin(input));
        min_d = std::min(min_d, sin(input));
        xi.push_back(input);
        d.push_back(sin(input));
        x.push_back(xi);
    }
    for (int i = 0; i<s; i++) {
        x[i][0] = (x[i][0] - min_input) / (max_input - min_input);
        d[i] = (d[i] - min_d) / (max_d - min_d);
        //std::cout << "d: " << d[i] << std::endl;
        //std::cout << "x: " << x[i][0] << std::endl;
    }
}

void NeuralNetwork::fit(std::vector<std::vector<double> >& x, std::vector<double>& d, int epoch, const double learning_rate) {
    int s = x.size();
    for (int i = 0; i<epoch; i++) {
        double error = 0;
        double y;
        for (int j = 0; j<s; j++) {
            y = propagate(x[j]);
            //backpropagate(x[j], y, d[j], learning_rate);
            std::cout << "Predicted/Label: " << y << " " << d[j] << std::endl;
            error += (d[j] - y) * (d[j] - y);
        }
        //std::cout << "Error: " << error << std::endl;
    }

}

double NeuralNetwork::activation(const double x) const{
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::predict(std::vector<double>& xi) {
    return propagate(xi);
}

void NeuralNetwork::addLayer(int neurons_number, std::string function_name, int input_dim) {
    if (input_dim != 0) { // for the first input layer
        Layer layer(input_dim, neurons_number, function_name);
        m_layers.push_back(layer);
    }
    else { // for stacked layer the input dim is the number of neurons in the previous layer
        input_dim = m_layers.back().getNeuronsNumber();
        Layer layer(input_dim, neurons_number, function_name);
        m_layers.push_back(layer);
    }
}

double NeuralNetwork::propagate(const std::vector<double>& input) {
    std::vector<double> x = input;
    std::vector<double> y;
    for (std::vector<Layer>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        y = it->computeOutput(x);
        x = y;
    }
    return y[0];
    // int n_layer = m_layers[1].size();
    // for (int j = 0; j < n_layer; j++) {
    //     double y1j = 0;
    //     for (int k = 0; k < n; k++) {
    //         y1j += m_layers[0][k] * xi[k];
    //     }
    //     o.push_back(activation(y1j));
    // }
    // double ys = 0;
    // for (int k = 0; k < n_layer; k++) {
    //     ys += m_layers[1][k] * o[k];
    // }
    // return activation(ys);
}

void NeuralNetwork::backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate) {

    // int n_layer = m_layers[1].size();
    // double delta2 = (y - di) * y * (1 - y);
    // std::vector<double> weights;
    //
    // for (int i = 0; i<n_layer; i++) {
    //     weights.push_back(m_layers[1][i]);
    //     m_layers[1][i] -= learning_rate * o[i] * delta2;
    // }
    //
    // double delta1;
    // for (int i = 0; i < n_layer; i++) {
    //     delta1 = delta2 * weights[i] * o[i] * (1 - o[i]);
    //     m_layers[0][i] -= learning_rate * xi[0] * delta1;
    // }

}
