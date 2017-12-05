#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<std::vector<double> >& x, std::vector<double>& d, int s);
    void generateData(std::vector<std::vector<double> >& x, std::vector<double>& d, int s);
    void fit(std::vector<std::vector<double> >& x, std::vector<double>& d, int epoch, const double learning_rate);
    double predict(std::vector<double>& xi);
    void addLayer(int size);

protected:
    std::vector<std::vector<double> > m_layers;
    int m_hidden_layers;
    std::vector<double> o;

    double activation(const double x) const;
    double propagate(const std::vector<double>& xi);
    void backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate);
};

#endif
