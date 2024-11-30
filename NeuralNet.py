import numpy as np

class NeuralNet:
  def __init__(self, layers, epochs=100, learn_rate=0.01, momentum=0.9, fun="relu", val_percentage=0.2):
    self.L = len(layers)
    self.n = layers.copy()
    self.epochs = epochs
    self.learn_rate = learn_rate
    self.momentum = momentum
    self.val_percentage = val_percentage

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
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.theta = []
    for lay in range(self.L):
      self.theta.append(np.zeros(layers[lay]))

    self.train_loss = []
    self.val_loss = []

  def fit(self, X, y):
    pass

  def predict(self, X):
    predictions = []
    for sample in X:
      self.forward_pass(sample)
      predictions.append(self.xi[-1])
    return np.array(predictions)


  def loss_epochs(self):
    return np.array(self.train_loss), np.array(self.val_loss)


  def forward_pass(self,x):
    self.xi[0] = x
    for i in range(1, self.L):
      z = np.dot(self.w[1], self.xi[1 - 1]) + self.theta[1]
      self.xi[1] = self.fact(z)


  def backward_pass(self,y):
    pass
  def compute_loss(self, X, y):
    pass


  def sigmoid(self, x):
    return 1/(1+np.exp(-x))

  def sigmoid_derivative(self, x):
    return x*(1-x)

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


layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
