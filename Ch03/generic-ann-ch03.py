import numpy
import matplotlib.pyplot

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def activation_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

x=0.1
target = 0.3
learning_rate = 0.5
w=numpy.random.rand()
print("Initial W : ", w)

network_error = []
predicted_output = []

old_err = 0
for k in range(80000):
    # Forward Pass
    y = w*x
    predicted = sigmoid(y)
    err = error(predicted, target)
    
    predicted_output.append(predicted)
    network_error.append(err)
    
    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    g2 = activation_sop_deriv(predicted)
    
    g3 = sop_w_deriv(x)
    
    grad = g3*g2*g1
    print(predicted)
    
    w = update_w(w, grad, learning_rate)

    old_err = err

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
