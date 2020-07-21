import numpy
import MLP

x = numpy.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

y = numpy.array([[0],
                 [1],
                 [1],
                 [0]])

network_architecture = [2]

trained_ann = MLP.MLP.train(x=x,
                    y=y, 
                    net_arch=network_architecture,
                    max_iter=500000,
                    learning_rate=1,
                    activation="sigmoid",
                    GD_type="batch",
                    debug=True)

print("\nTraining Time : ", trained_ann["training_time_sec"])
print("Number of Training Iterations : ", trained_ann["elapsed_iter"])
print("Network Architecture : ", trained_ann["net_arch"])
print("Network Error : ", trained_ann["network_error"])

predicted_output = MLP.MLP.predict(trained_ann, x)
print("\nPredicted Output(s) : ", predicted_output)
