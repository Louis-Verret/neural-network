#include "NeuralNetwork.h"
#include "Utils.h"
#include "ProgressBar.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include <string>

NeuralNetwork::NeuralNetwork() {
    m_optimizer = new SGD();
    m_C = new MSE();
}

NeuralNetwork::NeuralNetwork(Optimizer* optimizer, char const* cost_name, char const* metric_name) :
 m_optimizer(optimizer)
{
    if (strcmp(cost_name, "mean_squared_error") == 0) {
        m_C = new MSE();
    } else if (strcmp(cost_name, "cross_entropy") == 0) {
        m_C = new CrossEntropy();
    }

    if (strcmp(metric_name, "accuracy") == 0) {
        m_metric = new CategoricalAccuracy();
    }
}

NeuralNetwork::~NeuralNetwork() {
    delete m_C;
    delete m_optimizer;
    int nb = m_layers.size();
    for (int i=0; i<nb; i++) {
        delete m_layers[i];
    }
}

void NeuralNetwork::fit(MatrixCPU& x, MatrixCPU& y, int epoch, const int batch_size) {
    std::vector<Matrix> batches_x;
    std::vector<Matrix> batches_y;
    separateDataInBatches(x, y, batches_x, batches_y, batch_size);
    int nb_batches = batches_x.size();
    ProgressBar pb(30);
    for (int t = 0; t<epoch; t++) {
        double error = 0;
        double metric = 0;
        std::cout << "\nEpoch: " << t+1 << "/" << epoch << std::endl;
        for (int j = 0; j<nb_batches; j++) {

            /* The two essential methods for training: propagation and backpropagation */
            propagate(batches_x[j]);
            backpropagate(batches_y[j], batch_size, t+1);

            double batch_error = m_C->computeError(m_a.back(), batches_y[j]).sumElem()/batch_size;
            double batch_metric = 0;
            error += batch_error;
            if (m_metric != NULL) {
                batch_metric = m_metric->computeMetric(m_a.back(), batches_y[j]);
                metric += batch_metric;
            }
            float progress = (float)(j+1) / (float)nb_batches;
            pb.display(progress, batch_error, batch_metric);
        }
        std::cout << "\nMean Error: " << error/batches_x.size() << std::endl;
        if (m_metric != NULL) {
            std::cout << "Mean Metric: " << metric/batches_x.size() << std::endl;
        }
    }
    std::cout << "Predicted/Label: " << m_a.back() << " " << batches_y[nb_batches-1] << std::endl;

}

void NeuralNetwork::separateDataInBatches(MatrixCPU& x, MatrixCPU& y, std::vector<Matrix>& batches_x, std::vector<Matrix>& batches_y, const int batch_size) {
    for (int i = 0; i<x.getM(); i+=batch_size) {
        int bound = 0;
        if (i + batch_size < x.getM()) {
            bound = batch_size;
        } else {
            bound = x.getM() - i;
        }
        Matrix xi(x.getN(), bound);
        Matrix yi(y.getN(), bound);
        std::vector<double> vec_xi(xi.getPaddingM() * xi.getPaddingN());
        std::vector<double> vec_yi(yi.getPaddingM() * yi.getPaddingN());
        for (int k = 0; k < bound; k++) {
            for (int j = 0; j < x.getN(); j++) {
                vec_xi[j * xi.getPaddingM() + k] = x(j, i + k);
            }
            for (int j = 0; j<y.getN(); j++) {
                vec_yi[j * yi.getPaddingM() + k] = y(j, i + k);
            }
        }
        cl::Buffer buffer_xi = cl::Buffer(GPU::context, vec_xi.begin(), vec_xi.end(), true);
        xi.setBuffer(buffer_xi);
        batches_x.push_back(xi);

        cl::Buffer buffer_yi = cl::Buffer(GPU::context, vec_yi.begin(), vec_yi.end(), true);
        yi.setBuffer(buffer_yi);
        batches_y.push_back(yi);
    }
}

void NeuralNetwork::validate(Matrix& x, Matrix& y) {
    Matrix y_predict = predict(x);
    if (m_metric != NULL) {
        double metric = m_metric->computeMetric(y_predict, y);
        std::cout << "Metric: " << metric << std::endl;
    }
}

Matrix NeuralNetwork::predict(Matrix& xi) {
    propagate(xi);
    return m_a.back();
}

void NeuralNetwork::addLayer(int neurons_number, char const* function_name, int input_dim) {
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

void NeuralNetwork::propagate(Matrix& input) {
    m_a.clear();
    m_z.clear();
    m_a.push_back(input);
    for (std::vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        m_z.push_back((*it)->add((*it)->multiply(m_a.back())));
        m_a.push_back((*it)->activate(m_z.back()));
    }
}

void NeuralNetwork::backpropagate(const Matrix& y, const int batch_size, int epoch_num) {
    Matrix delta_suiv(y.getN(), y.getM());
    if (strcmp(m_C->getName(), "cross_entropy") == 0 && strcmp(m_layers.back()->getActivationFunction()->getName(), "softmax") == 0) {
        delta_suiv = m_a.back() - y;
    } else {
        Matrix z_curr = m_layers.back()->getActivationFunction()->evalDev(m_z.back());
        Matrix gradient = m_C->computeErrorGradient(m_a.back(), y);
        delta_suiv = gradient.hadamardProduct(z_curr);
    }
    int L = m_layers.size();
    Layer* layer = m_layers[L-1];
    m_optimizer->updateWeights(layer, m_a[L-1], delta_suiv, batch_size, epoch_num);
    m_optimizer->updateBias(layer, delta_suiv, batch_size, epoch_num);

    for(int l = L-2; l >= 0; l--) {
        Layer* layer = m_layers[l];
        Layer* layer_suiv = m_layers[l+1];
        Matrix delta_curr = (layer_suiv->getWeights().transpose() * delta_suiv).hadamardProduct(layer->getActivationFunction()->evalDev(m_z[l]));
        m_optimizer->updateWeights(layer, m_a[l], delta_curr, batch_size, epoch_num);
        m_optimizer->updateBias(layer, delta_curr, batch_size, epoch_num);
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
