#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>

#include "Layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    void fit(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int epoch, const double learning_rate);
    std::vector<double> predict(std::vector<double>& xi);
    void addLayer(int neurons_number, std::string function_name, int input_dim = 0);
    void save(const char* file_name);

    std::vector<Layer*> getLayers() const {return m_layers;};

protected:
    std::vector<Layer*> m_layers;
    std::vector<std::vector<double> > m_z;
    std::vector<std::vector<double> > m_a;
    int input_dim;
    std::vector<double> o;

    std::vector<double> computeGradient(const std::vector<double>& di);
    void propagate(const std::vector<double>& input);
    void backpropagate(const std::vector<double>& di, const double learning_rate);
};

std::ostream& operator << (std::ostream& out, const NeuralNetwork& net);

#endif
