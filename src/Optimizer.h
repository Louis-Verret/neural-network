#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"

class Optimizer
{
public:
    Optimizer();
    virtual ~Optimizer();

    virtual void updateWeights(Layer* layer, const Matrix& a, const Matrix& delta, int batch_size, int epoch_num) const = 0;
    virtual void updateBias(Layer* layer, const Matrix& delta, int batch_size, int epoch_num) const = 0;
};

class SGD : public  Optimizer
{
public:
    SGD(const double learning_rate = 0.01, const double momentum = 0.9, const double decay = 0);
    ~SGD();

    virtual void updateWeights(Layer* layer, const Matrix& a, const Matrix& delta, int batch_size, int epoch_num) const;
    virtual void updateBias(Layer* layer, const Matrix& delta, int batch_size, int epoch_num) const;

private:
    const double m_learning_rate;
    const double m_momentum;
    const double m_decay;
};

class Adam : public  Optimizer
{
public:
    Adam(const double learning_rate = 0.001, const double beta_1 = 0.9, const double beta_2 = 0.999, const double epsilon = 1e-8, const double decay = 0);
    ~Adam();

    virtual void updateWeights(Layer* layer, const Matrix& a, const Matrix& delta, int batch_size, int epoch_num) const;
    virtual void updateBias(Layer* layer, const Matrix& delta, int batch_size, int epoch_num) const;

private:
    const double m_learning_rate;
    const double m_beta_1;
    const double m_beta_2;
    const double m_epsilon;
    const double m_decay;
};




#endif
