import numpy as np

class NeuralNet:
  def __init__(self, layers, epochs=0, learn_rate=0, momentum=0, fun="relu", val_percentage=0):
    self.L = len(layers)
    self.n = layers.copy()
    activation_functions = ["sigmoid", "relu", "linear", "tanh"]
    if fun not in activation_functions:
      self.fact = "relu"
    else:
      self.fact = fun
    self.fact = fun
    self.xi = []
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

  def fit(self, X, y):
    pass
  def predict(self, X):
    pass
  def loss_epochs(self):
    pass

layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
