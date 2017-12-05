#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

class NeuralNetwork
{
public:
    void fit(std::vector<double>& x, double y);

protected:
    std::vector<double> m_weights;

    std::vector<double>& propagate(const std::vector<double>& x) const;
    void backpropagate(const std::vector<double>& x, const double y, const double d) const;
};

#endif
