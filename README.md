# Neural Networks C++ Library

This repository constitutes a high level and optimized C++ library allowing the construction of any kind of neural network architecture. The library uses OpenMP for parallel computing.  
A sample executable handling the MNIST-Handwritten Digit Recognition Problem is prodived

## MNIST

### Download the data

```
cd data  
wget https://pjreddie.com/media/files/mnist_train.csv  
wget https://pjreddie.com/media/files/mnist_test.csv
```

### Run the executable (after downloading the data)
```
cd ..  
mkdir build  
cd build  
cmake ..  
make  
./mnist
```

## Construct your own model
The library allows a programming user to construct a neural network architecture. For that, write a C++ main and follow the steps:  

1 - Define the optimizer and init the network
```
Optimizer* opti = new SGD(0.1, 0.9);  
NeuralNetwork net(opti, "mean_squared_error");
```
2 - Add layers:
```
net.addLayer(20, "tanh", 100); // input 100; hidden 20
net.addLayer(1, "linear"); // output 1
```
3 - Train the model
```
net.fit(x_train, y_train, 1, 128);
```

4 - Validate on other data
```
net.validate(x_test, y_test);
```
5 - Save the model
```
net.save("../data/mnist_model.data");
```

## Loading a pre-trained model
Two models are provided: sinus.model and mnist.model.  
To use them, write a C++ main and first declare a network:
```
Optimizer* opti = new SGD(0.1, 0.9);  
NeuralNetwork net(opti, "mean_squared_error");
```
Then, you can load a model:
```
net.load("../model/sinus.model");
```
