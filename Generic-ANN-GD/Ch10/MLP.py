import numpy
import itertools
import sys
import time

class MLP:
    trained_ann = {}

    def train(x, y, net_arch, max_iter=5000, tolerance=0.0000001, learning_rate=0.001, activation="sigmoid", GD_type="stochastic",
              debug=False):
        """
        train(x, y, w, max_iter, tolerance, learning_rate, activation="sigmoid", debug=False)
        Training artificial neural network (ANN) using gradient descent (GD).

        Inputs:
            x: Training data inputs.
            y: Training data outputs.
            net_arch: Network architecture defining number of inputs/outputs and number of neurons per each hidden layer.
            max_iter: Maximum number of iterations for training the network.
            tolerance: The desired minimum difference between predicted and target values. The network tries to reach this value if possible.
            learning_rate: Learning rate to adapt the network weights while learning.
            activation: Activation function to be used for building the hidden layers. Only one activation function can be used for all hidden layers.
            GD_type: Can be either stochastic or batch.
            debug: If True, informational messages are printed while training. Note that this increases the training time.

        Outputs:
            trained_ann: A dictionary representing the trained neural network. It holds all information necessary for restoring the trained network.
        """

        # Number of inputs, number of neurons per each hidden layer, number of output neurons
        # The network architecture defined in the net_arch list does not include the number of inputs nor the number of outputs. The lines below updates the net_arch list to include these 2 numbers.
        in_size = x.shape[1]
        if y.ndim == 1:
            out_size = 1
        else:
            out_size = y.shape[1]

        net_arch = [in_size] + net_arch  # Number of inputs is added at the beginning of the list.
        net_arch = net_arch + [out_size]  # Number of outputs is added at the end of the list.
        net_arch = numpy.array(net_arch)  # Converting net_arch from a list to a NumPy array.

        net_arch, w_initial, b_initial = MLP.initialize_weights(net_arch)
        w = w_initial.copy()
        b = b_initial.copy()

        start_time = time.time()

        network_error = sys.maxsize
        if (debug == True):
            print("Training Started")

        if activation == "sigmoid":
            activation_func = MLP.sigmoid
        elif activation == "relu":
            activation_func = MLP.relu
        else:
            activation_func = MLP.sigmoid
            print("Sorry. Only Sigmoid and ReLU are supported at the current time. Sigmoid will still be used.")

        # pred_target_diff holds the difference between the predicted output and the target output to be compared by the tolerance.
        pred_target_diff = sys.maxsize
        
        w_temp = []
        b_temp = []
        for k1 in range(len(w)):
            w_temp.append(numpy.zeros(w[k1].shape))
            b_temp.append(numpy.zeros(b[k1].shape))

        iter_num = 0
        while (iter_num < max_iter and pred_target_diff > tolerance):
            network_error = 0 # Declare a variable to hold the network error across all samples.
            pred_target_diff = 0 # Declare a variable to hold the difference between the targets and predictions across all samples.
            all_samples_predictions = []
            for sample_idx in range(x.shape[0]):
                curr_sample_x = x[sample_idx] # inputs of current sample.
                curr_sample_y = y[sample_idx] # Target of current sample.

                sop_activ_mul = MLP.forward_path(curr_sample_x, w, b, activation_func)

                prediction = sop_activ_mul[-1][-1] # Predictions for current sample.
                all_samples_predictions.append(prediction)

                sample_pred_target_diff = numpy.sum(numpy.abs(curr_sample_y - prediction))
                pred_target_diff = pred_target_diff + sample_pred_target_diff

                sample_error = MLP.error(prediction, curr_sample_y)
                network_error = network_error + sample_error

                if GD_type == "stochastic": # For stochastic gradient descent.
                    w, b = MLP.backward_pass(curr_sample_x, curr_sample_y, w, b, net_arch, sop_activ_mul, prediction, out_size, activation, learning_rate)
                elif GD_type=="batch": # For batch gradient descent.
                    w_sample, b_sample = MLP.backward_pass(curr_sample_x, curr_sample_y, w, b, net_arch, sop_activ_mul, prediction, out_size, activation, learning_rate)

                    for k1 in range(len(w)):
                        w_temp[k1] = w_temp[k1] + w_sample[k1]
                        b_temp[k1] = b_temp[k1] + b_sample[k1]
                else:
                    print("Sorry. Only stochastic and batch gradient descent are supported. Stochastic gradient descent will be used.")
                    w, b = MLP.backward_pass(curr_sample_x, curr_sample_y, w, b, net_arch, sop_activ_mul, prediction, out_size, activation, learning_rate)

            if GD_type == "batch":
                for k1 in range(len(w)):
                    w_temp[k1] = w_temp[k1]/x.shape[0]
                    b_temp[k1] = b_temp[k1]/x.shape[0]

                w = w_temp.copy()
                b = b_temp.copy()
    
                w_temp = []
                b_temp = []
                for k1 in range(len(w)):
                    w_temp.append(numpy.zeros(w[k1].shape))
                    b_temp.append(numpy.zeros(b[k1].shape))

            if (debug == True):
                print("\nIteration : ", iter_num, "\nError : ", network_error)
            iter_num = iter_num + 1

        if (debug == True):
            print("Training Finished")
        end_time = time.time()

        training_time = end_time - start_time

        MLP.trained_ann["w"] = w
        MLP.trained_ann["b"] = b
        MLP.trained_ann["max_iter"] = max_iter
        MLP.trained_ann[
            "elapsed_iter"] = iter_num
        MLP.trained_ann["tolerance"] = tolerance
        MLP.trained_ann["activation"] = activation
        MLP.trained_ann["learning_rate"] = learning_rate
        MLP.trained_ann["initial_w"] = w_initial
        MLP.trained_ann["net_arch"] = net_arch
        MLP.trained_ann["num_hidden_layers"] = net_arch.shape[0] - 2
        MLP.trained_ann["training_time_sec"] = training_time
        MLP.trained_ann["network_error"] = network_error
        MLP.trained_ann["in_size"] = in_size
        MLP.trained_ann["out_size"] = out_size

        return MLP.trained_ann

    def initialize_weights(net_arch):
        """
        initialize_weights(net_arch)
        Initializing the neural network weights.

        Inputs:
            net_arch: Network architecture defining number of inputs/outputs and number of neurons per each hidden layer. The user input might need some refine.

        Outputs:
            net_arch: The final refined network architecture [if neccessary].
            w: The initial weights.
            b: Initial bias values.
        """

        w = []
        w_temp = []
        b = []
        for layer_counter in numpy.arange(net_arch.shape[0] - 1):
            rand_array = numpy.random.uniform(low=0.0, high=0.1, size=net_arch[layer_counter + 1])
            b.append(rand_array)
            for neuron_counter in numpy.arange(net_arch[layer_counter + 1]):
                rand_array = numpy.random.uniform(low=0.0, high=0.1, size=net_arch[layer_counter])
                w_temp.append(rand_array)
            w_temp = numpy.array(w_temp)
            w.append(w_temp)
            w_temp = []
            
        return net_arch, w, b

    def forward_path(x, w, b, activation_func):
        """
        forward_path(x, w, activation_func)
        Implementation of the forward pass.

        Inputs:
            x: training inputs
            w: current set of weights.
            b: current bias values.
            activation_func: A string representing the activation function used.
        Outputs:
            sop_activ_mul: An array representing the outputs of each layer
                The sop_activ_mul array has number of rows equal to number of layers. For each layer, 2 things are calculated:
                    1) SOP between inputs and weights.
                    2) Activation output for SOP.
                For example, If there are 4 layers, then the shape of the sop_activ_mul array is (4, 2).
                sop_activ_mul[x][0]: SOP in layer x.
                sop_activ_mul[x][1]: Activation in layer x.
        """
        sop_activ_mul = []

        sop_activ_mul_temp = []
        curr_multiplicand = x
        for layer_num in range(len(w)):
            sop_temp = numpy.matmul(curr_multiplicand, w[layer_num].T) + b[layer_num]
            activ_temp = activation_func(sop_temp)

            curr_multiplicand = activ_temp

            sop_activ_mul_temp.append([sop_temp, activ_temp])
            sop_activ_mul.extend(sop_activ_mul_temp)
            sop_activ_mul_temp = []
        return sop_activ_mul

    def sigmoid(sop):
        """
        sigmoid(sop)
        Implementation of the sigmoid activation function.

        Inputs:
            sop: A single value representing the sum of products between the layer inputs and its weights.

        Outputs:
            A single value representing the output of the sigmoid.
        """
        return 1.0 / (1 + numpy.exp(-1 * sop))
    
    def relu(x):
        """
        relu(sop)
        Implementation of the rectified linear unit (ReLU) activation function.

        Inputs:
            sop: A single value representing the sum of products between the layer inputs and its weights.

        Outputs:
            A single value representing the output of the relu.
        """
        return numpy.maximum(0, x)

    def error(predicted, target):
        """
        error(predicted, target)
        Preduction error in the current iteration.

        Inputs:
            predicted: The predictions of the network using its current parameters.
            target: The correct outputs that the network should predict.
        Outputs:
            Error.
        """
        return numpy.power(predicted - target, 2)

    def backward_pass(x, y, w, b, net_arch, sop_activ_mul, prediction, num_outputs, activation, learning_rate):        
        """
        backward_pass(x, y, w, net_arch, sop_activ_mul, prediction, num_outputs, activation, learning_rate)
        Implementation of the backward pass for training the neural network using gradient descent.

        Inputs:
            x: Training inputs. Used for calcualting the derivative for the weights between the input layer and the hidden layer.
            y: Training data outputs.
            w: Current weights which are to be updated during the backward pass using the gradient descent.
            b: Bias values to be updated during the backward pass using the gradient descent.
            net_arch: A NumPy array defining the network archietcture defining the number of neurons in all layers. It is used here to know the number of neurons per each layer.
            sop_activ_mul: An array holding all calculations during the forward pass. This includes the sum of products between the inputs and weights of each layer and the results of the activation functions.
            prediction: The predicted outputs of the network using the current weights.
            num_outputs: Number of outputs per sample.
            learning_rate: Learning rate to adapt the network weights while learning.

        Outputs:
            w: The updated weights using gradient descent.
        """
        if activation == "sigmoid":
            activation_sop_deriv = MLP.sigmoid_sop_deriv
        elif activation == "relu":
            activation_sop_deriv = MLP.relu_sop_deriv
        else:
            activation_sop_deriv = MLP.sigmoid_sop_deriv
            print("Sorry. Only Sigmoid and ReLU are supported at the current time. Sigmoid will still be used.")

        g1 = MLP.error_predicted_deriv(prediction, y)

        g2 = activation_sop_deriv(sop_activ_mul[-1][0])

        output_layer_derivs = g2 * g1

        layer_weights_derivs = []
        layer_weights_grads = []

        if net_arch.shape[0] == 2:
            layer_weights_derivs = [MLP.sop_w_deriv(sop_activ_mul[-1][1])] + layer_weights_derivs
            layer_weights_grads = [layer_weights_derivs[0] * output_layer_derivs] + layer_weights_grads

            for out_neuron_idx in range(num_outputs):
                w[-1][out_neuron_idx] = w[-1][out_neuron_idx] - layer_weights_grads[0][out_neuron_idx] * learning_rate
                b[-1][out_neuron_idx] = b[-1][out_neuron_idx] - layer_weights_derivs[0] * output_layer_derivs[out_neuron_idx] * learning_rate

            MLP.trained_ann["derivative_chain"] = output_layer_derivs
            MLP.trained_ann["weights_derivatives"] = layer_weights_derivs
            MLP.trained_ann["weights_gradients"] = layer_weights_grads

            return w, b

        w_old = w.copy()

        layer_weights_derivs = [MLP.sop_w_deriv(sop_activ_mul[-1][1])] + layer_weights_derivs

        layer_weights_grads = [layer_weights_derivs[0] * output_layer_derivs] + layer_weights_grads

        # Updating the weights between the last hidden layer and the output neurons.
        for out_neuron_idx in range(num_outputs):
            w[-1][out_neuron_idx] = w[-1][out_neuron_idx] - layer_weights_grads[0][out_neuron_idx] * learning_rate
            b[-1][out_neuron_idx] = b[-1][out_neuron_idx] - layer_weights_derivs[0] * output_layer_derivs[out_neuron_idx] * learning_rate

        SOPs_ACTIVs_deriv_individual = []
        deriv_chain_final = []

        # Derivatives of the neurons between the last hidden layer and the output neurons.
        SOPs_ACTIVs_deriv_individual.append(output_layer_derivs)
        deriv_chain_final.append(output_layer_derivs)

        ############# GENERIC GRADIENT DESCENT #############
        # The idea is to prepare the individual derivatives of all neurons in all layers. These derivatives include:
        # -) Derivative of activation (output) to SOP (input)
        # -) Derivative of SOP (input) to activation (output)
        # These derivative not include the following:
        # -) Derivative of SOP (output) to the weight. It will be calculated later.
        # Using the chain rule, combinations are created from these individual derivatives.
        # X) The total derivative at a given neuron is the mean of products of all these combinations.
        # Y) For every neuron at a given layer, the derivative of the SOP (output) to the weight is calculated.
        # The gradient to update a given weight is the product between the mean (step X) and the weight derivative (step Y).

        # Derivatives of all other layers
        # If the network has no or just 1 hidden layer, this loop will not be executed. # It works when there are more than 1 hidden layer.
        curr_idx = -1
        for curr_lay_idx in range(len(w)-2, -1, -1):
            # ACTIVs_deriv are identical for all neurons in a given layer. If a layer has 5 neurons, then the length of ACTIVs_deriv is 5.
            # But SOPs_deriv returns 5 values for every neuron. Thus, there will be an array of 5 rows, one row for one neuron.
            SOPs_deriv = MLP.sop_w_deriv(w_old[curr_lay_idx + 1])
            ACTIVs_deriv = activation_sop_deriv(sop_activ_mul[curr_lay_idx][0])

            temp = []
            for curr_neuron_idx in range(net_arch[curr_idx]):
                temp.append(ACTIVs_deriv * SOPs_deriv[curr_neuron_idx])
            curr_idx = curr_idx - 1
            temp = numpy.array(temp)
            # Individual Derivatives of the Network Except the Weights Derivatives
            SOPs_ACTIVs_deriv_individual = [temp] + SOPs_ACTIVs_deriv_individual

            temp2 = MLP.deriv_chain_prod(temp[0], deriv_chain_final[-1])
            for neuron_calc_derivs_idx in range(1, temp.shape[0]):
                # Only last element in the deriv_chain_final array is used for calculating the chain.
                # This reuses the previous calculations because some chains were calculated previously.
                temp2 = temp2 + MLP.deriv_chain_prod(temp[neuron_calc_derivs_idx], deriv_chain_final[-1])
            deriv_chain_final.append(temp2)

            #### Calculate WEIGHTS Derivatives and Gradients
            # Index 1 means output of activation function for all neurons per layer.
            # At layer with index i, the derivs (layer_weights_derivs) and grads (layer_weights_grads) of its weights are at index i.
            # For example, layer indexed i=0 has its derivs (layer_weights_derivs) and grads (layer_weights_grads) in index 0.
            if curr_lay_idx == 0:
                layer_weights_derivs = [MLP.sop_w_deriv(x)] + layer_weights_derivs
            else:
                layer_weights_derivs = [MLP.sop_w_deriv(sop_activ_mul[curr_lay_idx - 1][1])] + layer_weights_derivs
            layer_weights_grads = [layer_weights_derivs[0]] + layer_weights_grads

        # Derivatives of the Entire Network (Except Weights Derivatives) Chains Following the Chain Rule.
        deriv_chain_final = numpy.array(deriv_chain_final)
        # Derivatives of the Entire Nework including the Weights Derivatives
        layer_weights_derivs = numpy.array(layer_weights_derivs)
        # Gradients of the Entire Network Weights
        layer_weights_grads = numpy.array(layer_weights_grads)

        deriv_chain_final = numpy.flip(deriv_chain_final)

        MLP.trained_ann["derivative_chain"] = deriv_chain_final
        MLP.trained_ann["weights_derivatives"] = layer_weights_derivs
        MLP.trained_ann["weights_gradients"] = layer_weights_grads

        #### Updating Weights of All Layers Except Last Layer Because it was Updated Previously
        for layer_idx in range(len(w) - 1):
            w[layer_idx] = MLP.update_weights(w[layer_idx], layer_weights_grads[layer_idx],
                                              deriv_chain_final[layer_idx], learning_rate)
            b[layer_idx] = MLP.update_bias(b[layer_idx], deriv_chain_final[layer_idx], learning_rate)
        return w, b

    def error_predicted_deriv(predicted, target):
        """
        error_predicted_deriv(predicted, target)
        Derivative of the error to the predicted value.

        Inputs:
            predicted: The predictions of the network using its current parameters.
            target: The correct outputs that the network should predict.

        Outputs:
            Derivative of the error to the predicted value.
        """
        return 2 * (predicted - target)

    def sigmoid_sop_deriv(sop):
        """
        sigmoid_sop_deriv(sop)
        Calculating the derivative of the sigmoid to the sum of products.

        Inputs:
            sop: Sum of products.

        Outputs:
            Derivative of the sum of products to sigmoid.
        """
        return MLP.sigmoid(sop) * (1.0 - MLP.sigmoid(sop))

    def relu_sop_deriv(sop):
        """
        relu_sop_deriv(sop)
        Calculating the derivative of the relu to the sum of products.

        Inputs:
            sop: Sum of products.

        Outputs:
            Derivative of the sum of products to relu.
        """
        sop[sop <= 0] = 0.0
        sop[sop > 0] = 1.0
        return sop

    def sop_w_deriv(x):
        """
        sop_w_deriv(x)
        Derivative of the sum of products to the weights.

        Inputs:
            x: inputs to the current layer.

        Outputs:
            Derivative of the sum of products to the weights.
        """
        return x

    def deriv_chain_prod(layer_derivs, previous_lay_derivs_chains):
        """
        deriv_chain_prod(derivs_arrs)
        Derivative chains of a given layer.
        
        Inputs:
            layer_derivs: Derivatives of the current layer.
            previous_lay_derivs_chains: Derivatives chains of the previous layer.
        Outputs:
            deriv_chain_prod_sum: A NumPy array representing the sum of the derivatives products in all chains.
        """
        derivs_arrs = [layer_derivs] + [[previous_lay_derivs_chains]]
        deriv_combinations = numpy.array(list(itertools.product(*derivs_arrs)))
        num_products_in_layer = deriv_combinations.shape[0]
        num_neurons_in_layer = len(derivs_arrs[0])
        num_chains_for_neuron = numpy.uint32(num_products_in_layer / num_neurons_in_layer)

        #        print("\n# of Products in Layer : ", num_products_in_layer,
        #              "\n# of Neurons in Layer : ", num_neurons_in_layer,
        #              "\n# of Products per Neuron : ", num_chains_for_neuron)

        deriv_chain_prod_sum = []
        for neuron_idx in range(num_neurons_in_layer):
            start_idx = neuron_idx * num_chains_for_neuron
            end_idx = (neuron_idx + 1) * num_chains_for_neuron
            deriv_chain = deriv_combinations[start_idx:end_idx]
            deriv_chain_prod = numpy.prod(deriv_chain, axis=1)
            # When there are more than 1 chain to reach a neuron, the sum of all chains is calculated and returned as a single value.
            deriv_chain_prod_sum.append(numpy.concatenate(deriv_chain_prod).sum())
            # Update weight according to chains product sum (deriv_chain_prod_sum)
        deriv_chain_prod_sum = numpy.array(deriv_chain_prod_sum)
        return deriv_chain_prod_sum

    def update_weights(weights, layer_weights_grads, deriv_chain_final, learning_rate):
        """
        update_weights(weights, gradients, learning_rate)
        Updating the weights based on the calcualted gradients.

        Inputs:
            weights: Weights of a layer tbe updated.
            layer_weights_grads: Gradient of the current layer for updating its weights.
            deriv_chain_final: Chains of derivatives from the layers higher than the current layer.
            learning_rate: Learnign rate.

        Outputs:
            Updated weights.
        """
        for neuron_idx in range(weights.shape[0]):
            weights[neuron_idx] = weights[neuron_idx] - layer_weights_grads * deriv_chain_final[
                neuron_idx] * learning_rate
        return weights

    def update_bias(b, deriv_chain_final, learning_rate):
        """
        update_weights(weights, gradients, learning_rate)
        Updating the weights based on the calcualted gradients.

        Inputs:
            b: Bias.
            deriv_chain_final: Chains of derivatives from the layers higher than the current layer.
            learning_rate: Learnign rate.

        Outputs:
            Updated b.
        """
        for neuron_idx in range(b.shape[0]):
            b[neuron_idx] = b[neuron_idx] - deriv_chain_final[neuron_idx] * learning_rate
        return b

    def predict(trained_ann, x):
        """
        predict(trained_ann, x)
        Making prediction for a new sample.
        
        Inputs:
            trained_ann: A dictionary representing trained MLP.
            x: The new sample.
        Outputs:
            prediction: The predicted output for the current sample.
        """
        in_size = trained_ann["in_size"]
        if x.ndim == 1:
            if x.shape[0] != in_size:
                print("Input shape ", x.shape[0], " does not match the expected network input shape ", in_size)
                return
        else:
            if x.shape[1] != in_size:
                print("Input shape ", x.shape[1], " does not match the expected network input shape ", in_size)
                return

        w = trained_ann["w"]
        b = trained_ann["b"]

        activation = trained_ann["activation"]
        if activation == "sigmoid":
            activation_func = MLP.sigmoid
        elif activation == "relu":
            activation_func = MLP.relu
        else:
            activation_func = MLP.sigmoid
            print("Sorry. Only Sigmoid and ReLU are supported at the current time. Sigmoid will still be used.")

        sop_activ_mul = MLP.forward_path(x, w, b, activation_func)
        prediction = sop_activ_mul[-1][1]

        return prediction