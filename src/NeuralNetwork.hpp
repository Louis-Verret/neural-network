#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

class NeuralNetwork
{
public:
    void fit(std::vector<double> x, std::vector<double> y);

private:
    std::vector<std::vector<double> > weights_list;
};

#endif
