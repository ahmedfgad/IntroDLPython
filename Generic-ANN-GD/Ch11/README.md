# Part 8

Welcome to the project that builds a generic neural network trained using the gradient descent algorithm.

The implementation of this part ends by building a network that works supports:

1. Any number of outputs and not just limited to a single output.
2. Any number of samples and not just limited to a single sample.
3. Work with bias in both forward and backward passes.
4. Allow stochastic and batch modes for the gradient descent.

The script named `MLP.py` holds a class named `MLP` with all necessary methods and functions to build and network.

# Example

The `generic-ann-ch11.py` script has an example of using the the `MLP` class.

```python
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

