#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

class NeuralNetwork
{
public:
    NeuralNetwork(int n);
    void fit(std::vector<std::vector<double> >& x, std::vector<double>& d, int epoch, const double learning_rate);
    double predict(std::vector<double>& xi);

protected:
    std::vector<double> m_weights;
    double propagate(const std::vector<double>& xi) const;
    void backpropagate(const std::vector<double>& xi, const double y, const double di, const double learning_rate);
};

#endif
