# Work with Any Number of Hidden Layers

Welcome to the project that builds a generic neural network trained using the gradient descent algorithm.

The implementation of this part ends by building a network that works supports:

1. Use of any number of inputs.
2. Use of any number of hidden layers 
3. Use of any number of neurons in the hidden layers
4. Just a single output. 
5. A single training sample.

The code is organized by creating a script named `MLP.py` that holds a class named `MLP` with all necessary methods and functions to build and network.

# Example

The `generic-ann-ch09.py` script has an example of using the the `MLP` class.

```python
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
```

## Book: [Practical Computer Vision Applications Using Deep Learning with CNNs](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

[![kivy-book](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)

# Contact Us

- E-mail: [ahmed.f.gad@gmail.com](mailto:ahmed.f.gad@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
- [Amazon Author Page](https://amazon.com/author/ahmedgad)
- [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
- [Paperspace](https://blog.paperspace.com/author/ahmed)
- [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
- [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
- [GitHub](http://github.com/ahmedfgad)

