# Building a Generic Python Implementation for a Neural Network Trained using the Gradient Descent

This project builds a generic Python implementation of a neural network trained using the gradient descent algorithm. The implementation starts from scratch using NumPy. 

Here is a quick summary of the project directories:

- **Ch04** builds and trains the simplest neural network with just a single input neuron and a single output neuron. The network does not have any hidden layers. There is just a single training sample.
- **Ch05** extends **Ch04** implementation to allow the network to work with any number of inputs. 
- **Ch06** introduces a single hidden layer with just 2 hidden neurons.
- **Ch07** is just different from **Ch06** by using any number of hidden neurons within a single hidden layer. 
- **Ch08** uses 2 hidden layers with any number of hidden neurons. 
- **Ch09** adds an additional hidden layer to **Ch08** so that there are 3 hidden layers with any number of hidden neurons.
- **Ch10** is a generic implementation that allows the network to work with any number of hidden layers and any number of neurons within such layers. After the network is trained, all network parameters are saved. Such parameters can be loaded later for making predictions.
- **Ch11** adds more generalization as the network can work with any number of samples and any number of outputs.

At the moment, only **Ch04** to **Ch07** are made public to the project. The other directories will be made available soon.