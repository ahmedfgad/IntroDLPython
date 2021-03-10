import numpy
import matplotlib.pyplot

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

x = numpy.array([0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6])
target = numpy.array([0.2])

learning_rate = 0.001

# Number of inputs, number of neurons per each hidden layer, number of output neurons (which must be 1 at this time)
network_architecture = numpy.array([x.shape[0], 5, 1])

# Initializing the weights of the entire network
w = []
w_temp = []
for layer_counter in numpy.arange(network_architecture.shape[0]-1):
    for neuron_nounter in numpy.arange(network_architecture[layer_counter+1]):
        w_temp.append(numpy.random.rand(network_architecture[layer_counter]))
    w.append(numpy.array(w_temp))
    w_temp = []
w = numpy.array(w)
w_old = w
print("Initial W : ", w)

network_error = []
predicted_output = []

old_err = 0
for k in range(50000):
    # Forward Pass
    # Hidden Layer Calculations
    sop_hidden = numpy.matmul(w[0], x)

    sig_hidden = sigmoid(sop_hidden)

    # Output Layer Calculations
    sop_output = numpy.sum(w[1][0]*sig_hidden)

    predicted = sigmoid(sop_output)
    err = error(predicted, target)

    predicted_output.append(predicted)
    network_error.append(err)
    
    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    ### Working with weights between hidden and output layer
    g2 = sigmoid_sop_deriv(sop_output)
    g3 = sop_w_deriv(sig_hidden)
    grad_hidden_output = g3*g2*g1
    w[1][0] = update_w(w[1][0], grad_hidden_output, learning_rate)
    
    ### Working with weights between input and hidden layer
    g5 = sop_w_deriv(x)
    for neuron_idx in numpy.arange(w[0].shape[0]):
        g3 = sop_w_deriv(w_old[1][0][neuron_idx])
        g4 = sigmoid_sop_deriv(sop_hidden[neuron_idx])
        grad_hidden_input = g5*g4*g3*g2*g1
        w[0][neuron_idx] = update_w(w[0][neuron_idx], grad_hidden_input, learning_rate)

    w_old = w
    print(predicted)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(network_error)
matplotlib.pyplot.title("Iteration Number vs Error")
matplotlib.pyplot.xlabel("Iteration Number")
matplotlib.pyplot.ylabel("Error")

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(predicted_output)
matplotlib.pyplot.title("Iteration Number vs Prediction")
matplotlib.pyplot.xlabel("Iteration Number")
matplotlib.pyplot.ylabel("Prediction")
