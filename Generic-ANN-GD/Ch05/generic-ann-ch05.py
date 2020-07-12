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

x1=0.1
x2=0.4
x3=1.1
x4=1.3
x5=1.8
x6=2.0
x7=0.01
x8=0.9
x9=0.8
x10=1.6

target = 0.7
learning_rate = 0.001

w1=numpy.random.rand()
w2=numpy.random.rand()
w3=numpy.random.rand()
w4=numpy.random.rand()
w5=numpy.random.rand()
w6=numpy.random.rand()
w7=numpy.random.rand()
w8=numpy.random.rand()
w9=numpy.random.rand()
w10=numpy.random.rand()

print("Initial W : ", w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)

predicted_output = []
network_error = []

old_err = 0
for k in range(80000):
    # Forward Pass
    y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w8*x8 + w9*x9 + w10*x10
    predicted = sigmoid(y)
    err = error(predicted, target)
    
    predicted_output.append(predicted)
    network_error.append(err)

    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    g2 = sigmoid_sop_deriv(y)
    
    g3w1 = sop_w_deriv(x1)
    g3w2 = sop_w_deriv(x2)
    g3w3 = sop_w_deriv(x3)
    g3w4 = sop_w_deriv(x4)
    g3w5 = sop_w_deriv(x5)
    g3w6 = sop_w_deriv(x6)
    g3w7 = sop_w_deriv(x7)
    g3w8 = sop_w_deriv(x8)
    g3w9 = sop_w_deriv(x9)
    g3w10 = sop_w_deriv(x10)
    
    gradw1 = g3w1*g2*g1
    gradw2 = g3w2*g2*g1
    gradw3 = g3w3*g2*g1
    gradw4 = g3w4*g2*g1
    gradw5 = g3w5*g2*g1
    gradw6 = g3w6*g2*g1
    gradw7 = g3w7*g2*g1
    gradw8 = g3w8*g2*g1
    gradw9 = g3w9*g2*g1
    gradw10 = g3w10*g2*g1
    
    w1 = update_w(w1, gradw1, learning_rate)
    w2 = update_w(w2, gradw2, learning_rate)
    w3 = update_w(w3, gradw3, learning_rate)
    w4 = update_w(w4, gradw4, learning_rate)
    w5 = update_w(w5, gradw5, learning_rate)
    w6 = update_w(w6, gradw6, learning_rate)
    w7 = update_w(w7, gradw7, learning_rate)
    w8 = update_w(w8, gradw8, learning_rate)
    w9 = update_w(w9, gradw9, learning_rate)
    w10 = update_w(w10, gradw10, learning_rate)

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
