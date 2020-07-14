import numpy
import matplotlib.pyplot

def sigmoid(sop):
    return 1.0 / (1 + numpy.exp(-1 * sop))

def error(predicted, target):
    return numpy.power(predicted - target, 2)

def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate * grad

x = numpy.array([0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6])
target = numpy.array([0.2])

learning_rate = 0.001

# Number of inputs, number of neurons per each hidden layer, number of output neurons
network_architecture = numpy.array([x.shape[0], 8, 4, 1])

# Initializing the weights of the entire network
w = []
w_temp = []
for layer_counter in numpy.arange(network_architecture.shape[0] - 1):
    for neuron_nounter in numpy.arange(network_architecture[layer_counter + 1]):
        w_temp.append(numpy.random.rand(network_architecture[layer_counter]))
    w.append(numpy.array(w_temp))
    w_temp = []
w = numpy.array(w)
w[-1] = w[-1][0]
w_old = w

predicted_output = []
network_error = []

layer_idx = 0
for k in range(80000):
    # Forward Pass
    # First Hidden Layer Calculations
    sop_hidden1 = numpy.matmul(w[layer_idx], x)

    sig_hidden1 = sigmoid(sop_hidden1)

    # Second Hidden Layer Calculations
    layer_idx = layer_idx + 1  # =0+1=1
    sop_hidden2 = numpy.matmul(w[layer_idx], sig_hidden1)

    sig_hidden2 = sigmoid(sop_hidden2)

    # Output Layer Calculations
    layer_idx = layer_idx + 1  # =1+1=2
    sop_output = numpy.sum(w[layer_idx] * sig_hidden2)

    predicted = sigmoid(sop_output)
    err = error(predicted, target)

    predicted_output.append(predicted)
    network_error.append(err)

    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)  # shape=(1 Value)

    ### Working with weights between second hidden and output layer
    g2 = sigmoid_sop_deriv(sop_output)  # shape=(1 Value)

    g3 = sop_w_deriv(sig_hidden2)

    grad_hidden_output = g3 * g2 * g1

    w[layer_idx] = update_w(w[layer_idx], grad_hidden_output, learning_rate)

    ### Working with weights between first hidden and second hidden layer
    g3 = sop_w_deriv(w_old[layer_idx])
    g4 = sigmoid_sop_deriv(sop_hidden2)
    g5 = sop_w_deriv(sig_hidden1)

    layer_idx = layer_idx - 1  # =2-1=1

    for neuron_idx in numpy.arange(g3.shape[0]):
        grad_hidden_input = g5 * g4[neuron_idx] * g3[neuron_idx] * g2 * g1
        w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], grad_hidden_input, learning_rate)

    ### Working with weights between first hidden and input layer
    g6 = sigmoid_sop_deriv(sop_hidden1)
    g7 = sop_w_deriv(x)

    layer_idx = layer_idx - 1  # =1-1=0

    for neuron_idx1 in numpy.arange(g3.shape[0]):
        g5 = sop_w_deriv(w_old[layer_idx + 1][neuron_idx1])
        for neuron_idx in numpy.arange(g5.shape[0]):
            grad_hidden_input = g7 * g6[neuron_idx] * g5[neuron_idx] * g4[neuron_idx1] * g3[neuron_idx1] * g2 * g1
            w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], grad_hidden_input, learning_rate)

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
