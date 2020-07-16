import numpy
import matplotlib.pyplot

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

x = numpy.array([0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6])
target = numpy.array([0.45])

learning_rate = 0.001

# Number of inputs, number of neurons per each hidden layer, number of output neurons
network_architecture = numpy.array([x.shape[0], 8, 5, 3, 1])

# Initializing the weights of the entire network
w = []
w_temp = []
for layer_counter in numpy.arange(network_architecture.shape[0]-1):
    for neuron_nounter in numpy.arange(network_architecture[layer_counter+1]):
        w_temp.append(numpy.random.rand(network_architecture[layer_counter]))
    w.append(numpy.array(w_temp))
    w_temp = []
w = numpy.array(w)
w[-1] = w[-1][0] # Last set of weigts are of shape (1, n). For simplicity, first dim is removed.
w_old = w

predicted_output = []
network_error = []

#sigmoid_sop_deriv = relu_sop_deriv
layer_idx = 0
for k in range(80000):
    ### Forward Pass
    
    sop_activ_mul = []
    sop_activ_mul_temp = []
    curr_multiplicand = x
    for layer_num in range(w.shape[0]):
        sop_temp = numpy.matmul(w[layer_num], curr_multiplicand)
        activ_temp = sigmoid(sop_temp)
        curr_multiplicand = activ_temp

        sop_activ_mul_temp.append([sop_temp, activ_temp])
        sop_activ_mul.extend(sop_activ_mul_temp)
        sop_activ_mul_temp = []
    sop_activ_mul = numpy.array(sop_activ_mul)

    # Error Calculations
    err = error(sop_activ_mul[3][1], target)

    predicted_output.append(sop_activ_mul[3][1])
    network_error.append(err)

    ### Backward Pass
    layer_idx = w.shape[0]-1 # layer_idx=3
    
    # Derivative of error to predicted (sig4-output)
    g1 = error_predicted_deriv(sop_activ_mul[3][1], target) # shape=(1 Value)

    ### Working with weights between third hidden and output layer
    # Derivative of predicted (sig4-output) to (sop4-output)
    g2 = sigmoid_sop_deriv(sop_activ_mul[3][0]) # shape=(1 Value)

    # Derivative of (sop4-output) to weights between third hidden & output layer
    g3 = sop_w_deriv(sop_activ_mul[2][1])

    # Gradients for updating weights between third hidden & output layer
    grad_hidden_output = g3*g2*g1

    # Updating weights between third hidden and output layer
    w[layer_idx] = update_w(w[layer_idx], grad_hidden_output, learning_rate)

    ### Working with weights between second hidden and third hidden layer
    # Derivative of (sop4-output) to sig3
    g3 = sop_w_deriv(w_old[layer_idx])
    # Derivative of sig3 to sop3
    g4 = sigmoid_sop_deriv(sop_activ_mul[2][0])
    # Derivative of sop3 to weights between second hidden and third hidden layer
    g5 = sop_w_deriv(sop_activ_mul[1][1])

    layer_idx = layer_idx - 1 # =3-1=2

    for neuron_idx in numpy.arange(g3.shape[0]):
        # Gradients for updating weights between second hidden and third hidden layer
        grad_hidden_input = g5*g4[neuron_idx]*g3[neuron_idx]*g2*g1
        # Updating weights between second hidden and third hidden layer
        w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], grad_hidden_input, learning_rate)

    ### Working with weights between first hidden and second hidden layer
    # Derivative of sig2 to sop2
    g6 = sigmoid_sop_deriv(sop_activ_mul[1][0])
    # Derivative of sop2 to weights between first hidden and second hidden layer
    g7 = sop_w_deriv(sop_activ_mul[0][1])

    layer_idx = layer_idx - 1 # =2-1=1

    for neuron_idx1 in numpy.arange(g3.shape[0]): # =g4.shape[0]
        # Derivative of sop3 to sig2
        g5 = sop_w_deriv(w_old[layer_idx+1][neuron_idx1])
        for neuron_idx in numpy.arange(g5.shape[0]):
            # Gradients for updating weights between first hidden and second hidden layer
            grad_hidden_input = g7*g6[neuron_idx]*g5[neuron_idx]*g4[neuron_idx1]*g3[neuron_idx1]*g2*g1
            # Updating weights between first hidden and second hidden layer
            w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], grad_hidden_input, learning_rate)

    ### Working with weights between input layer and first hidden
    # Derivative of sig1 to sop1
    g8 = sigmoid_sop_deriv(sop_activ_mul[0][0])
    # Derivative of sop1 to inputs (x)
    g9 = sop_w_deriv(x)

    layer_idx = layer_idx - 1 # =1-1=0

    for neuron_idx2 in numpy.arange(g3.shape[0]): # =g4.shape[0]
        for neuron_idx1 in numpy.arange(g5.shape[0]): # =g6.shape[0]
            # Derivative of sop2 to sig1
            g7 = sop_w_deriv(w_old[layer_idx+1][neuron_idx1])
            for neuron_idx in numpy.arange(g7.shape[0]):
                # Gradients for updating weights between input layer and first hidden layer
                grad_hidden_input = g9*g8[neuron_idx]*g7[neuron_idx]*g6[neuron_idx1]*g5[neuron_idx1]*g4[neuron_idx2]*g3[neuron_idx2]*g2*g1
                # Updating updating weights between input layer and first hidden layer
                w[layer_idx][neuron_idx] = update_w(w[layer_idx][neuron_idx], grad_hidden_input, learning_rate)

    w_old = w
    print(sop_activ_mul[3][1])

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
