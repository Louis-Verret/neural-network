#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

class NeuralNetwork
{
public:
    void fit(std::vector<double> x, std::vector<double> y);

protected:
    std::vector<double> m_weights;
    double propagate(const std::vector<double>& x) const;
    void backpropagate(const std::vector<double>& y) const;
};

#endif
