#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>


NeuralNetwork::NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int s) {
    generateData(x, d, s);
}

NeuralNetwork::~NeuralNetwork() {
    int nb = m_layers.size();
    for (int i=0; i<nb; i++) {
        delete m_layers[i];
    }
}

void NeuralNetwork::generateData(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int s) {
    srand(time(NULL));
    double max_input = -10;
    double min_input = 10;
    double max_d = -1;
    double min_d = 1;
    for (int i = 0; i<s; i++) {
        std::vector<double> xi;
        std::vector<double> di;
        double input = ((double)rand() / (double)RAND_MAX) * 6.28 - 3.14;
        max_input = std::max(max_input, input);
        min_input = std::min(min_input, input);
        max_d = std::max(max_d, sin(input));
        min_d = std::min(min_d, sin(input));
        xi.push_back(input);
        di.push_back(sin(input));
        d.push_back(di);
        x.push_back(xi);
    }
    for (int i = 0; i<s; i++) {
        x[i][0] = (x[i][0] - min_input) / (max_input - min_input);
        d[i][0] = (d[i][0] - min_d) / (max_d - min_d);
        //std::cout << "d: " << d[i] << std::endl;
        //std::cout << "x: " << x[i][0] << std::endl;
    }
}

void NeuralNetwork::fit(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int epoch, const double learning_rate) {
    int s = x.size();
    for (int i = 0; i<epoch; i++) {
        double error = 0;
        for (int j = 0; j<s; j++) {
            propagate(x[j]);
            backpropagate(d[j], learning_rate);
            //std::cout << "Predicted/Label: " << y << " " << d[j] << std::endl;
            error += (d[j][0] - m_a.back()[0]) * (d[j][0] -  m_a.back()[0]);
        }
        std::cout << "Error: " << error << std::endl;
    }
    std::cout << "Predicted/Label: " << m_a.back()[0] << " " << d[s-1][0] << std::endl;

}

std::vector<double> NeuralNetwork::predict(std::vector<double>& xi) {
    propagate(xi);
    return m_a.back();
}

void NeuralNetwork::addLayer(int neurons_number, std::string function_name, int input_dim) {
    if (input_dim != 0) { // for the first input layer
        //std::cout << "Creating first layer" << std::endl;
        Layer* layer = new Layer(input_dim, neurons_number, function_name);
        m_layers.push_back(layer);
    }
    else { // for stacked layer the input dim is the number of neurons in the previous layer
        //std::cout << "Adding layer" << std::endl;
        input_dim = m_layers.back()->getNeuronsNumber();
        Layer* layer = new Layer(input_dim, neurons_number, function_name);
        m_layers.push_back(layer);
    }
}

void NeuralNetwork::propagate(const std::vector<double>& input) {
    //std::vector<double> x = input;
    //std::vector<double> y;
    m_a.clear();
    m_z.clear();
    m_a.push_back(input);
    for (std::vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        // std::cout << it->getNeuronsNumber() << std::endl;
        // std::cout << it->getInputDim() << std::endl;
        //y = it->computeOutput(x);
        //x = y;
        m_z.push_back((*it)->multiply(m_a.back()));
        m_a.push_back((*it)->activate(m_z.back()));
    }
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

std::vector<double> NeuralNetwork::computeGradient(const std::vector<double>& di) {
    std::vector<double> gradient;
    std::vector<double> a_L = m_a.back();
    for (int j = 0; j<di.size(); j++) {
        gradient.push_back(a_L[j] - di[j]);
    }
    return gradient;
}

std::vector<double> operator*(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        perror("Invalid size for Hadamard product");
    }
    std::vector<double> res;
    for (int i = 0; i<v1.size(); i++) {
        res.push_back(v1[i] * v2[i]);
    }
    return res;
}

void NeuralNetwork::backpropagate(const std::vector<double>& di, const double learning_rate) {

    std::vector<double> z_curr = m_layers.back()->getActivationFunction()->evalDev(m_z.back());
    std::vector<double> gradient = computeGradient(di);
    //std::cout << "size: " << gradient.size() << " " << z_curr.size() << std::endl;
    std::vector<double> delta_suiv = gradient * z_curr;
    std::vector<double> delta_curr;
    int L = m_layers.size();
    Layer* layer = m_layers[L-1];
    layer->updateWeights(m_a[L-1], delta_suiv, learning_rate);

    for(int l = L-2; l >= 0; l--) {
        Layer* layer = m_layers[l];
        Layer* layer_suiv = m_layers[l+1];
        //std::cout << "size M init: " << layer->getWeights()->getN() << " " << layer->getWeights()->getM()  << std::endl;
        //std::cout << "size: " << layer->getWeights()->transpose().getM() << " " << delta_suiv.size() << std::endl;
        delta_curr = (layer_suiv->getWeights()->transpose() * delta_suiv) * layer->getActivationFunction()->evalDev(m_z[l]);
        layer->updateWeights(m_a[l], delta_curr, learning_rate);
        delta_suiv = delta_curr;
    }

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


std::ostream& operator << (std::ostream& out, const NeuralNetwork& net) {
    std::vector<Layer*> layers = net.getLayers();
    for (std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); ++it) {
        out << *(*it) << std::endl;
        out << "----------------------------------------------------" << std::endl;
    }
    return out;
}
