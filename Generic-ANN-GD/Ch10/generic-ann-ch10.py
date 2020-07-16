import MLP
import numpy

x = numpy.array([0.1, 0.4, 4.1])
y = numpy.array([0.2])

# network_architecture defines the number of hidden neurons in the hidden layers. It must be a list not any other datatype.
network_architecture = [7, 5, 4]

# Network Parameters
trained_ann = MLP.MLP.train(x=x,
                    y=y,
                    net_arch=network_architecture,
                    max_iter=500,
                    learning_rate=0.7,
                    debug=True)

print("Derivative Chains : ", trained_ann["derivative_chain"])
print("Training Time : ", trained_ann["training_time_sec"])
print("Number of Training Iterations : ", trained_ann["elapsed_iter"])

predicted_output = MLP.MLP.predict(trained_ann, numpy.array([0.2, 3.1, 1.7]))
print("Predicted Output : ", predicted_output)
