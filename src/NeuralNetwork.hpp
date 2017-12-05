#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

class NeuralNetwork
{
public:
    void fit(std::vector<double> x, std::vector<double> y);

protected:
    std::vector<std::vector<double> > m_weights_list;

    std::vector<double>& propagate(const std::vector<double>& x) const;
};

#endif
