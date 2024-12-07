import pickle

import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    """
    A fully connected feedforward neural network.

    :param layers: List of integers representing the number of neurons in each layer.
    :type layers: list[int]
    :param epochs: Number of training iterations (default: 200).
    :type epochs: int, optional
    :param learn_rate: Learning rate for gradient descent (default: 0.001).
    :type learn_rate: float, optional
    :param momentum: Momentum factor for gradient updates (default: 0.9).
    :type momentum: float, optional
    :param fun: Activation function to use ('sigmoid', 'relu', 'linear', or 'tanh') (default: "tanh").
    :type fun: str, optional
    :param val_percentage: Percentage of data reserved for validation during training (default: 0.2).
    :type val_percentage: float, optional
    :param save: Flag to save the model (default: False).
    :type save: bool, optional

    **Attributes**:

    - **L** (*int*): Number of layers in the network.
    - **n** (*list[int]*): Number of neurons in each layer.
    - **save** (*bool*): Flag to save the model (default: False).
    - **epochs** (*int*): Number of training iterations.
    - **learn_rate** (*float*): Learning rate for weight updates.
    - **momentum** (*float*): Momentum factor for gradient updates.
    - **val_percentage** (*float*): Validation data percentage.
    - **fun** (*str*): Chosen activation function.
    - **activation_functions** (*dict*): Map of activation function names to their implementations.
    - **derivatives** (*dict*): Map of activation function names to their derivatives.
    - **w** (*list[np.ndarray]*): Weights of the network layers.
    - **theta** (*list[np.ndarray]*): Thresholds (biases) of the network layers.
    - **train_loss** (*list[float]*): Training loss history.
    - **val_loss** (*list[float]*): Validation loss history.

    **Methods**:

    - **fit(X, y)**: Train the network with given input-output data.
    - **forward_pass(x)**: Perform a forward pass of input data.
    - **backward_pass(y)**: Backpropagate the error through the network.
    - **update_weights()**: Update weights and thresholds using gradients.
    - **loss_epochs()**: Return the training and validation loss histories.
    - **compute_loss(X, y)**: Compute the quadratic error for given inputs and outputs.
    - **predict(X)**: Generate predictions for new input data.
    - **save_model(file_name)**: Save the network's parameters to a file.
    - **load_model(file_name)**: Load a previously saved network from a file.
    - **plot_loss()**: Plot training and validation loss histories.

    **Activation Functions**:

    - **sigmoid(x)**: Compute the sigmoid activation.
    - **sigmoid_derivative(x)**: Compute the derivative of the sigmoid function.
    - **relu(x)**: Compute the ReLU activation.
    - **relu_derivative(x)**: Compute the derivative of the ReLU function.
    - **linear(x)**: Compute the linear activation.
    - **linear_derivative(x)**: Compute the derivative of the linear function.
    - **tanh(x)**: Compute the hyperbolic tangent activation.
    - **tanh_derivative(x)**: Compute the derivative of the hyperbolic tangent function.

    **Example**::

        # Define a neural network with 3 layers
        net = NeuralNet(layers=[4, 5, 1], epochs=100, learn_rate=0.01, fun="relu")

        # Train the network on input data X and target values y
        net.fit(X, y)

        # Predict outputs for new inputs
        predictions = net.predict(new_X)
    """

    def __init__(self, layers, epochs=100, learn_rate=0.001, momentum=0.9, fun="tanh", val_percentage=0.2, save=False):
        self.L = len(layers)
        self.n = layers.copy()
        self.save = save
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.val_percentage = val_percentage
        self.fun = fun
        self.activation_functions = {"sigmoid": self.sigmoid, "relu": self.relu, "linear": self.linear,
                                     "tanh": self.tanh}
        self.derivatives = {"sigmoid": self.sigmoid_derivative, "relu": self.relu_derivative,
                            "linear": self.linear_derivative, "tanh": self.tanh_derivative}

        if fun not in self.activation_functions:
            self.fact = self.activation_functions["relu"]
            self.fact_derivatives = self.derivatives["relu"]
        else:
            self.fact = self.activation_functions[fun]
            self.fact_derivatives = self.derivatives[fun]

        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        self.w = []
        self.w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2. / layers[lay - 1]))

        self.theta = []
        self.delta = []
        for lay in range(self.L):
            self.theta.append(np.random.randn(layers[lay]) * np.sqrt(2. / layers[lay]))
            self.delta.append(np.zeros(layers[lay]))

        self.h = []
        for lay in range(self.L):
            self.h.append(np.zeros((layers[lay], 1)))

        self.d_w = []
        self.d_w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((self.n[lay], self.n[lay - 1])))

        self.d_theta = []
        for lay in range(self.L):
            self.d_theta.append(np.zeros(self.n[lay]))

        self.d_w_prev = []
        self.d_w_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((self.n[lay], self.n[lay - 1])))

        self.d_theta_prev = []
        for lay in range(self.L):
            self.d_theta_prev.append(np.zeros(self.n[lay]))

        self.train_loss = []
        self.val_loss = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
       Train the neural network using the provided input data and labels.

       This method splits the dataset into training and validation sets based on the
       `val_percentage` parameter, and then performs the training loop for the specified
       number of epochs. During each epoch, the model performs feed-forward propagation,
       back-propagates the error, and updates the weights and thresholds. The training and
       validation errors are computed and stored at each epoch.

       :param X: Input data, with shape (num_samples, num_features).
       :type X: numpy.ndarray
       :param y: Target output data, with shape (num_samples, num_output_units).
       :type y: numpy.ndarray
       """
        num_val = int(self.val_percentage * X.shape[0])
        X_train, X_val = X[num_val:], X[:num_val]
        y_train, y_val = y[num_val:], y[:num_val]
        num_patterns = X_train.shape[0]

        for epoch in range(self.epochs):
            # Shuffle the training data
            indices = np.random.permutation(num_patterns)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Training patterns loop
            for pat in range(num_patterns):
                # Step 5: Choose a random pattern
                x_mu = X_shuffled[pat]
                z_mu = y_shuffled[pat]

                # Step 6: Feed-forward propagation
                self.forward_pass(x_mu)

                # Step 7: Back-propagate the error
                self.backward_pass(z_mu)

                # Step 8: Update weights and thresholds
                self.update_weights()

            # Step 9: Calculate prediction quadratic error for training set
            train_error = self.compute_loss(X_train, y_train)
            # Step 10: Calculate prediction quadratic error for validation set
            val_error = self.compute_loss(X_val, y_val)

            # Store errors for plotting
            self.train_loss.append(train_error)
            self.val_loss.append(val_error)

            # Optional: Print epoch progress
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Error: {train_error:.4f}, Validation Error: {val_error:.4f}")
        self.plot_loss()

        if self.save:
            self.save_model(f"models/model_{self.fun}_{self.epochs}_{self.learn_rate}_{self.momentum}_{self.n}.pkl")

    def forward_pass(self, x: np.ndarray) -> None:
        """
          Perform a forward pass through the network.

          This method computes the field (h) and activations (xi) for each layer in the network
          starting from the input layer up to the output layer. It uses the weights and thresholds
          to compute the activations based on the chosen activation function.

          :param x: Input data for the forward pass.
          :type x: numpy.ndarray
      """
        # ξ(1) = x in this case xi[0]
        self.xi[0] = x
        # Loop through each layer starting from the first hidden layer
        for l in range(1, self.L):
            # Compute field h for the current layer
            # h(ℓ)_i = sum ω(ℓ)_ij * ξ(ℓ−1)_j − θ(ℓ)
            h = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.h[l] = h  # Store field, l-1 bc of the dimensionality of h
            # Apply activation function to compute the activations for the current layer
            # ξ(ℓ)_i = g(h(ℓ)_i)
            self.xi[l] = self.fact(h)

    def backward_pass(self, y: np.ndarray) -> None:
        """
         Perform backpropagation of errors through the network.

         This method calculates the error for the output layer and then propagates the error
         backwards through the hidden layers. The errors (deltas) are stored for each layer
         and will be used for updating the weights and thresholds.

         :param y: The true target output for the current pattern.
         :type y: numpy.ndarray
    """
        # Step 1: Compute the output layer error (delta)
        # ∆(L)_i = g′(h(L)_i) * (o_i(x) − z_i) and o(x) = ξ(L) which is xi[-1]
        self.delta[-1] = (self.xi[-1] - y) * self.fact_derivatives(self.h[-1])

        # Step 2: Backpropagate errors through the hidden layers
        for l in range(self.L - 2, 0, -1):  # From second-to-last layer to first hidden layer
            # ∆(ℓ−1)_j = g′(h(ℓ−1)_j) * sum (∆(ℓ)_i * ω(ℓ)_ij)
            self.delta[l] = self.fact_derivatives(self.h[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])

    def update_weights(self) -> None:
        """
          Update the weights and thresholds using the calculated deltas.

          This method applies the gradient descent algorithm with momentum to update the weights
          and thresholds of the network, using the errors from the backpropagation step. The
          previous updates are stored to implement the momentum.

    """
        for l in range(1, self.L):
            # Update weights with momentum
            self.d_w[l] = -self.learn_rate * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
            self.d_w_prev[l] = self.d_w[l]
            self.w[l] += self.d_w[l]

            # Update thresholds with momentum
            self.d_theta[l] = self.learn_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]
            self.d_theta_prev[l] = self.d_theta[l]
            self.theta[l] += self.d_theta[l]

    def loss_epochs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the training and validation loss histories.

        This method returns two arrays containing the training and validation loss values
        for each epoch, which can be used for monitoring the model's performance.

        :return: Tuple containing two numpy arrays: training loss and validation loss.
        :rtype: tuple
    """
        return np.array(self.train_loss), np.array(self.val_loss)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
          Compute the quadratic error (loss) for a given dataset.

          This method computes the squared error between the network's predictions and the
          true values for each sample in the dataset, and returns the average error.

          :param X: Input data, with shape (num_samples, num_features).
          :type X: numpy.ndarray
          :param y: True output data, with shape (num_samples, num_output_units).
          :type y: numpy.ndarray
          :return: The computed loss (mean squared error).
          :rtype: float
    """
        total_error = 0
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            total_error += np.sum((self.xi[-1] - y[i]) ** 2)
        return total_error / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
         Generate predictions for the given input data.

         This method performs a forward pass for each sample in the input data and returns
         the predicted output based on the trained weights and thresholds.

         :param X: Input data for which predictions are to be made.
         :type X: numpy.ndarray
         :return: Predictions generated by the network.
         :rtype: numpy.ndarray
    """
        predictions = []
        for sample in X:
            self.forward_pass(sample)
            predictions.append(self.xi[-1])
        return np.array(predictions)

    # ACTIVATION FUNCTIONS

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation function.

        The sigmoid function maps input values to a range between 0 and 1. It is commonly used
        in neural networks to introduce non-linearity.

        :param x: Input data for the activation function.
        :type x: numpy.ndarray
        :return: The sigmoid activation of the input.
        :rtype: numpy.ndarray
    """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
         Compute the derivative of the sigmoid activation function.

         The derivative of the sigmoid function is used during backpropagation to adjust weights
         in the network. It is computed as sigmoid(x) * (1 - sigmoid(x)).

         :param x: Input data for the activation function derivative.
         :type x: numpy.ndarray
         :return: The derivative of the sigmoid activation for the input.
         :rtype: numpy.ndarray
    """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the ReLU (Rectified Linear Unit) activation function.

        The ReLU function is a widely used activation function that outputs 0 for negative inputs
        and returns the input value for positive inputs.

        :param x: Input data for the activation function.
        :type x: numpy.ndarray
        :return: The ReLU activation of the input.
        :rtype: numpy.ndarray
    """
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
         Compute the derivative of the ReLU activation function.

         The derivative of ReLU is used during backpropagation. It is 0 for negative inputs
         and 1 for positive inputs.

         :param x: Input data for the activation function derivative.
         :type x: numpy.ndarray
         :return: The derivative of the ReLU activation for the input.
         :rtype: numpy.ndarray
    """
        return (x > 0).astype(int)

    def linear(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the linear activation function.

        The linear activation function returns the input value itself, typically used in the output layer
        of a regression model.

        :param x: Input data for the activation function.
        :type x: numpy.ndarray
        :return: The linear activation of the input.
        :rtype: numpy.ndarray
    """
        return x

    def linear_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the linear activation function.

        The derivative of the linear function is always 1, as the output is a direct function of the input.

        :param x: Input data for the activation function derivative.
        :type x: numpy.ndarray
        :return: The derivative of the linear activation for the input.
        :rtype: numpy.ndarray
    """
        return np.ones_like(x)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """
         Compute the tanh (hyperbolic tangent) activation function.

         The tanh function maps input values to a range between -1 and 1, and is commonly used in
         neural networks to introduce non-linearity.

         :param x: Input data for the activation function.
         :type x: numpy.ndarray
         :return: The tanh activation of the input.
         :rtype: numpy.ndarray
    """
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """
         Compute the derivative of the tanh activation function.

         The derivative of the tanh function is used during backpropagation. It is calculated as
         1 - tanh(x)^2.

         :param x: Input data for the activation function derivative.
         :type x: numpy.ndarray
         :return: The derivative of the tanh activation for the input.
         :rtype: numpy.ndarray
    """
        return 1 - np.tanh(x) ** 2

    # SAVE / LOAD MODEL

    def save_model(self, file_name: str) -> None:
        """
      Save the trained neural network model to a file.

      This method serializes the model's parameters and saves them to a file using pickle.
      It stores the number of layers, layer structure, training parameters, and weights.

      :param file_name: The name of the file to save the model.
      :type file_name: str
      :raises: IOError if the file cannot be opened for writing.
      """
        model_data = {
            'num_layers': self.L,
            'layers': self.n,
            'epochs': self.epochs,
            'learn_rate': self.learn_rate,
            'momentum': self.momentum,
            'val_percentage': self.val_percentage,
            'fact': self.fact,
            'fact_derivatives': self.fact_derivatives,
            'w': self.w,
            'theta': self.theta,
            'd_w_prev': self.d_w_prev,
            'd_theta_prev': self.d_theta_prev,
        }
        with open(file_name, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model saved to {file_name}.")

    def load_model(self, file_name: str) -> None:
        """
         Load the neural network model from a file.

         This method deserializes the model's parameters from a file and restores them. It
         loads the number of layers, layer structure, training parameters, and weights.

         :param file_name: The name of the file to load the model from.
         :type file_name: str
         :raises: IOError if the file cannot be opened or pickle cannot load the data.
    """
        with open(file_name, 'rb') as file:
            model_data = pickle.load(file)

        # Restore model parameters
        self.L = model_data['num_layers']
        self.n = model_data['layers']
        self.epochs = model_data['epochs']
        self.learn_rate = model_data['learn_rate']
        self.momentum = model_data['momentum']
        self.val_percentage = model_data['val_percentage']
        self.fact = model_data['fact']
        self.fact_derivatives = model_data['fact_derivatives']
        self.w = model_data['w']
        self.theta = model_data['theta']
        self.d_w_prev = model_data['d_w_prev']
        self.d_theta_prev = model_data['d_theta_prev']

        print(f"Model loaded from {file_name}.")

    # PLOTTING

    def plot_loss(self):
        """
        Plot the training and validation loss after the training process.

        This method visualizes the training loss and validation loss over epochs,
        showing how the model improves with time and helps diagnose issues such as overfitting.

        :raises: ValueError if training or validation loss cannot be retrieved.
       """

        # Create the figure and axis for plotting
        fig = plt.figure()
        ax = fig.add_subplot()

        # Get the training and validation loss from the loss_epochs method
        train_loss, val_loss = self.loss_epochs()

        # Plot the training and validation losses
        ax.plot(train_loss, label='Training Error')
        if val_loss is not None:
            ax.plot(val_loss, label='Validation Error')

        # Add labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(
            f'Training and Validation Loss \n(learn_rate={self.learn_rate}, mom={self.momentum}, fact={self.fun}, layer_struc={self.n})')

        # Add legend
        ax.legend()
        fig.savefig(f"plots/plot_{self.fun}_{self.epochs}_{self.learn_rate}_{self.momentum}_{self.n}.png", format='png', bbox_inches='tight')
        print(f"Plot saved to plots/plot_{self.fun}_{self.epochs}_{self.learn_rate}.png")
        # Redraw the plot
        fig.canvas.draw()


