#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"

#include <iostream>


int main(int argc, char **argv) {
    Matrix x;
    Matrix y;
    readCSV("../data/reg_lin_train.csv", x, y);

    //generateSinusData(x, y, 100);

    Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    //Optimizer* opti = new SGD(0.0001, 0);

    NeuralNetwork net(opti, "mean_squared_error");

    net.addLayer(1, "linear", 1);
    //net.addLayer(1, "relu");

    net.fit(x, y, 1000, 10);

    //std::cout << net;
    // //net->save("../data/sinus_training.data");
    // double input = 1.57; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(pi/2): " << 2*output(0,0) -1 << std::endl;

    return 0;
}
