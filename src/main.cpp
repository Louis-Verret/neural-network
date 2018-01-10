#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"

#include <iostream>


int main(int argc, char **argv) {
    Matrix x;
    Matrix y;
    readCSV("../data/mock_data.csv", false, x, y);

    std::cout << x << std::endl;
    std::cout << x.argmax() << std::endl;
    // generateSinusData(x, y, 100);

    // Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    // Optimizer* opti = new SGD(1, 0.9);
    //
    // NeuralNetwork net(opti, "mean_squared_error");
    //
    // net.addLayer(5, "sigmoid", 1);
    // net.addLayer(5, "sigmoid");
    // net.addLayer(1, "sigmoid");
    //
    // // net.load("../data/sinus_training.data");
    //
    // net.fit(x, y, 10000, 20);
    //
    // net.save("../data/sinus_training.data");
    // double input = -1.57/3; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    // std::cout << net << std::endl;

    return 0;
}
