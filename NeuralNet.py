import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# TODO if learn rate is 0.01 or 0.1 the loss is always the same
class NeuralNet:
  def __init__(self, layers, epochs=200, learn_rate=0.001, momentum=0.9, fun="tanh", val_percentage=0.2):
    self.L = len(layers)
    self.n = layers.copy()
    self.epochs = epochs
    self.learn_rate = learn_rate
    self.momentum = momentum
    self.val_percentage = val_percentage
    self.fun = fun
    self.activation_functions = {"sigmoid": self.sigmoid, "relu": self.relu, "linear": self.linear, "tanh": self.tanh}
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

  def fit(self, X, y):
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

    self.save_model(f"model_{self.fun}_{self.epochs}_{self.learn_rate}.pkl")
  def update_weights(self):
    for l in range(1, self.L):  # but  then the first layer is not updated??
      # Update weights with momentum
      self.d_w[l] = -self.learn_rate * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
      self.d_w_prev[l] = self.d_w[l]
      self.w[l] += self.d_w[l]

      # Update thresholds with momentum
      self.d_theta[l] = self.learn_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]
      self.d_theta_prev[l] = self.d_theta[l]
      self.theta[l] += self.d_theta[l]

  def plot_loss(self):
    # Get the training and validation loss from the loss_epochs method
    train_loss, val_loss = self.loss_epochs()

    # Plot the training and validation errors
    plt.plot(train_loss, label='Training Error')
    if val_loss is not None:
      plt.plot(val_loss, label='Validation Error')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    # Add legend
    plt.legend()

    # Show the plot
    plt.pause(0.1)  # This makes the plot update in real-time
  def predict(self, X):
    predictions = []
    for sample in X:
      self.forward_pass(sample)
      predictions.append(self.xi[-1])
    return np.array(predictions)


  def loss_epochs(self):
    return np.array(self.train_loss), np.array(self.val_loss)


  def forward_pass(self,x):
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

  def backward_pass(self,y):
    # Step 1: Compute the output layer error (delta)
    # ∆(L)_i = g′(h(L)_i) * (o_i(x) − z_i) and o(x) = ξ(L) which is xi[-1]
    self.delta[-1] = (self.xi[-1] - y) * self.fact_derivatives(self.h[-1])

    # Step 2: Backpropagate errors through the hidden layers
    for l in range(self.L - 2, 0, -1):  # From second-to-last layer to first hidden layer
      # ∆(ℓ−1)_j = g′(h(ℓ−1)_j) * sum (∆(ℓ)_i * ω(ℓ)_ij)
      self.delta[l] = self.fact_derivatives(self.h[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])
  def compute_loss(self, X, y):
    total_error = 0
    for i in range(X.shape[0]):
      self.forward_pass(X[i])
      total_error += np.sum((self.xi[-1] - y[i]) ** 2)
    return total_error / X.shape[0]

  def sigmoid(self, x):
    return 1/(1+np.exp(-x))

  def sigmoid_derivative(self, x):
    return self.sigmoid(x) * (1-self.sigmoid(x))

  def relu(self, x):
    return np.maximum(0, x)

  def relu_derivative(self, x):
    return (x > 0).astype(int)

  def linear(self, x):
    return x

  def linear_derivative(self, x):
    return np.ones_like(x)

  def tanh(self, x):
    return np.tanh(x)

  def tanh_derivative(self, x):
    return 1- np.tanh(x) ** 2

  def save_model(self, file_name):
      """
      Save the neural network model to a file.
      :param file_name: The name of the file to save the model.
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

  def load_model(self, file_name):
    """
    Load the neural network model from a file.
    :param file_name: The name of the file to load the model from.
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

layers = [27, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[1], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")


# Load your dataset
df = pd.read_csv('clean.csv')

# Separate the features (X) and target (y)
X = df.drop(columns=['price']).values
y = df['price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a scaler for the target variable
y_scaler = MinMaxScaler()

# Reshape y to be a 2D array (required for scalers)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Now, train the neural network using the scaled data
nn.fit(X_train_scaled, y_train_scaled)
#nn.load_model('model.pkl')
y_predicted  = nn.predict(X_test_scaled)
y_predicted_descaled = y_scaler.inverse_transform(y_predicted)
# Assuming y_predicted and y_test_scaled are numpy arrays or lists
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predicted_descaled, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction')
plt.xlabel('Actual Scaled Prices (y_test_scaled)')
plt.ylabel('Predicted Scaled Prices (y_predicted)')
plt.title('Comparison of Predicted vs Actual Prices')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

print(
  'mean_squared_error : ', mean_squared_error(y_test.flatten(), y_predicted_descaled.flatten()))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test.flatten(), y_predicted_descaled.flatten()))
print(r2_score(y_test, y_predicted_descaled))
