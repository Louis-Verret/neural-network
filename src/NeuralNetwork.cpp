#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
    int nb = m_layers.size();
    for (int i=0; i<nb; i++) {
        delete m_layers[i];
    }
}

void NeuralNetwork::fit(Matrix& x, Matrix& d, int epoch, const double learning_rate, const int batch_size, double momentum) {
    std::vector<Matrix> batches_x;
    std::vector<Matrix> batches_d;
    separateDataInBatches(x, d, batches_x, batches_d, batch_size);
    int nb_batches = batches_x.size();
    for (int i = 0; i<epoch; i++) {
        double error = 0;
        for (int j = 0; j<nb_batches; j++) {
            propagate(batches_x[j]);
            backpropagate(batches_d[j], learning_rate, batch_size, momentum);

            Matrix diff = m_a.back() - batches_d[j];
            error += diff.hadamardProduct(diff).sumElem();
        }
        std::cout << i+1 << " Error: " << error/batch_size << std::endl;
    }
    std::cout << "Predicted/Label: " << m_a.back() << " " << batches_d[nb_batches-1] << std::endl;

}

void NeuralNetwork::separateDataInBatches(Matrix& x, Matrix& d, std::vector<Matrix>& batches_x, std::vector<Matrix>& batches_d, const int batch_size) {
    for (int i = 0; i<x.getM(); i+=batch_size) {
        int bound = 0;
        if (i + batch_size < x.getM()) {
            bound = batch_size;
        } else {
            bound = x.getM() - i;
        }
        Matrix xi(x.getN(), bound);
        Matrix di(d.getN(), bound);
        for (int k = 0; k<bound; k++) {
            for (int j = 0; j<x.getN(); j++) {
                xi(j, k) = x(j, i + k);
                di(j, k) = d(j, i + k);
            }
        }
        batches_x.push_back(xi);
        batches_d.push_back(di);
    }
}

Matrix NeuralNetwork::predict(Matrix& xi) {
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

void NeuralNetwork::save(const char* file_name) {
    std::ofstream file;
    file.open(file_name);
    if (file.is_open()) {
        file << m_layers.size() << std::endl;
        for (std::vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
            file << (*it)->getInputDim() << " " << (*it)->getNeuronsNumber() << std::endl;

        }
        file.close();
    }
}

void NeuralNetwork::propagate(Matrix& input) {
    m_a.clear();
    m_z.clear();
    m_a.push_back(input);
    for (std::vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        m_z.push_back((*it)->add((*it)->multiply(m_a.back())));
        m_a.push_back((*it)->activate(m_z.back()));
    }
}

Matrix NeuralNetwork::computeGradient(const Matrix& d) {
    Matrix a_L = m_a.back();
    return a_L - d;
}

void NeuralNetwork::backpropagate(const Matrix& d, const double learning_rate, int batch_size, double momentum) {
    Matrix z_curr = m_layers.back()->getActivationFunction()->evalDev(m_z.back());
    Matrix gradient = computeGradient(d);
    Matrix delta_suiv = gradient.hadamardProduct(z_curr);
    int L = m_layers.size();
    Layer* layer = m_layers[L-1];
    layer->updateWeights(m_a[L-1], delta_suiv, learning_rate, batch_size, momentum);
    layer->updateBias(delta_suiv, learning_rate, batch_size, momentum);

    for(int l = L-2; l >= 0; l--) {
        Layer* layer = m_layers[l];
        Layer* layer_suiv = m_layers[l+1];
        Matrix delta_curr = (layer_suiv->getWeights()->transpose() * delta_suiv).hadamardProduct(layer->getActivationFunction()->evalDev(m_z[l]));
        layer->updateWeights(m_a[l], delta_curr, learning_rate, batch_size, momentum);
        layer->updateBias(delta_curr, learning_rate, batch_size, momentum);
        delta_suiv = delta_curr;
    }
}


std::ostream& operator << (std::ostream& out, const NeuralNetwork& net) {
    std::vector<Layer*> layers = net.getLayers();
    for (std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); ++it) {
        out << *(*it) << std::endl;
        out << "----------------------------------------------------" << std::endl;
    }
    return out;
}
