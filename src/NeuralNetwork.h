#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

#include "Layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<double>& d, int s);
    void generateData(std::vector<std::vector<double> >& x, std::vector<double>& d, int s);
    void fit(std::vector<std::vector<double> >& x, std::vector<double>& d, int epoch, const double learning_rate);
    double predict(std::vector<double>& xi);
    void addLayer(int neurons_number, std::string function_name, int input_dim = 0);

protected:
    std::vector<Layer> m_layers;
    int input_dim;
    std::vector<double> o;

    double activation(const double x) const;
    double propagate(const std::vector<double>& xi);
    void backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate);
};

#endif
