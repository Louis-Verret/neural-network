# Neural networks C++ library

This repository constitutes a high level and optimized library using OpenMP allowing an user to construct any kind of neural network architecture.   
Two executables are provided : mainSinus and mainMNIST

## MNIST

This executable constructs a neural networks for hand written digits recognition. The architecture of the network is detailed in the report.

### To download the data
```
cd data  
wget https://pjreddie.com/media/files/mnist_train.csv  
wget https://pjreddie.com/media/files/mnist_test.csv
```

### Running the executable (after downloading the data)
```
cd ..  
mkdir build  
cd build  
cmake ..  
make  
./mnist
```

## Sinus
This executable constructs a neural networks for sinus function approximation. The data are generated thanks to a build-in function.

### Running the executable
```
./sinus
```

## Construct your own model
The library allows an user to construct a neural network architecture. For that, write a C++ main and follow the steps:  

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
