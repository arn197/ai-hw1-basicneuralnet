import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt

class TwoLayerMLP(object):

  def __init__(self, arch, std=1e-4):
    """
    Weights - random values
    Biases - set to zero

    Inputs:
    - arch: array containing number of nodes in each layer
    - std: standard deviation used for initializing weights
    """
    self.params = {}
    self.params['weights'] = []
    self.params['biases'] = []
    self.size = len(arch) - 1
    self.arch = arch
    for i in range(1, len(arch)):
        self.params['weights'].append(std * np.random.randn(arch[i - 1], arch[i]))
        self.params['biases'].append(np.zeros(arch[i]))

  def forward(self, X):
    """
    Runs forward propagation on input

    Parameters
    ----------
    Inputs
    X - Input samples, shape - (N, D)

    Outputs
    acts - List containing the activation array of each layer. Activation array
    contains the output values of each node in a layer
    """
    z = X
    activations = []
    for i in range(self.size):
        #Select weights and bias between each layer
        W, b = self.params['weights'][i], self.params['biases'][i]

        #Forward prop - done as matrix mult and adding bias at the end
        z = np.dot(z, W) + b
        if i == self.size - 1:
            activations.append(z) 
            # If the current output value corresponds to the output layer,
            # don't apply sigmoid to it, softmax will be applied later
            break
        z = sp.expit(z)
        if i < self.size - 1:
            activations.append(z)
    return activations

  def get_loss(self, X, y, scores, reg):
    """
    Calculates the loss according to the outputs and inputs

    Parameters
    ----------
    Inputs
    X - input matrix, shape - (N, D)
    y - ground truth array, shape - (N, 1)
    scores - the last column of activation array, corresponds to outputs at output layer
    reg - regularization term

    Outputs
    loss - the loss value calculated using cross entropy
    P - array containing softmax values of outputs
    """
    N, D = X.shape
    A = np.max(scores, axis=1)
    F = np.exp(scores - A.reshape(N, 1))
    softmax_activation = F / np.sum(F, axis=1).reshape(N, 1)
    loss = np.mean(-np.log(softmax_activation[range(y.shape[0]), y]))

    for i in range(self.size):
        W = self.params['weights'][i]
        loss += 0.5 * reg * np.sum(W * W)
    return loss, softmax_activation
  
  def backprop(self, X, y, softmax_activation, activations, reg):
    """
    Backprop function to compute gradients based on loss and layer activations

    Parameters
    ----------
    Inputs
    X - input array, shape - (N, D)
    y - ground truth array, shape - (N, 1)
    P - softmax activation of output layer
    acts - activations of each layer
    reg - regularization term

    Outputs
    grads - dictionary containing keys weights and biases, each of which contains the gradients per layer
    """
    W1, b1 = self.params['weights'][0], self.params['biases'][0]
    W2, b2 = self.params['weights'][1], self.params['biases'][1]
    _, C = W2.shape
    N, D = X.shape
    grads = {}
    

    # output layer
    y_1hot = np.zeros((N,C))
    for i in range(N):
        y_1hot[i,y[i]] = 1
    dEx = softmax_activation - y_1hot # partial derivative of loss w.r.t output
    dW = [0] * self.size # weight gradients array
    dB = [0] * self.size # bias gradients array

    # partial derivative of loss w.r.t weight matrix between hidden layer and output layer
    dW[-1] = np.dot(activations[-1].T, dEx)/N
    # partial derivative of loss w.r.t output node bias
    dB[-1] = np.mean(dEx, axis=0)
    
    activations = [X] + activations
    for i in range(self.size - 2, -1, -1):
        W = self.params['weights'][i + 1]
        dEy = np.dot(dEx, W.T)
        dEx = (activations[i + 1]*(1 - activations[i + 1])) * dEy
        # partial derivative of loss w.r.t weight matrix between hidden layer and output layer
        dW[i] = np.dot(activations[i].T, dEx)/activations[i].shape[0]
        dB[i] = np.mean(dEx, axis = 0)
    grads['weights'] = dW
    grads['biases'] = dB

    return grads

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_epochs=10, verbose=False):
    """
    Train this neural network using batch gradient descent.

    Inputs
    X: Training samples, shape - (N, D)
    y: Training labels, shape - (N, 1)
    learning_rate: Scalar giving learning rate for optimization
    reg: Scalar giving regularization strength
    num_epochs: Number of epochs
    verbose: boolean; if true print progress during optimization

    Outputs
    {
        loss_history : history of loss values for each epoch
        train_acc_history: history of training accuracy for each epoch
    }
    """

    loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):

        activations = self.forward(X)
        loss, softmax_activation = self.get_loss(X, y, activations[-1], reg)
        grads = self.backprop(X, y, softmax_activation, activations[:-1], reg)
        loss_history.append(loss)

        for i in range(self.size):
            self.params['weights'][i] -= grads['weights'][i] * learning_rate
            self.params['biases'][i] -= grads['biases'][i] * learning_rate

        train_acc = (self.predict(X) == y).mean()
        train_acc_history.append(train_acc)
        if verbose:
            print('Epoch %d: loss %f, train_acc %f'%(
                epoch+1, loss, train_acc))

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
    }


  def predict(self, X):
    """
    Returns the predictions according to the current state of the model

    Parameters
    ----------
    Inputs:
    - X: Numpy array of shape (N, D)

    Outputs:
    - y_pred: Numpy array of shape (N, 1) containing predictions of each sample in X
    """

    y_pred = self.forward(X)[-1]
    y_pred = np.exp(y_pred)/np.exp(np.sum(y_pred,axis=1)).reshape(-1,1)
    y_pred =  np.argmax(y_pred,axis=1)
    return y_pred

def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    
    Parameters
    ----------
    YTrue : ground truth
    YPredict : predicted values

    Returns
    cm : Confusion matrix
    accuracy : The accuracy of the predictions

    """
    len_labels = len(np.unique(YTrue))
    cm = np.zeros((len_labels ,len_labels ), int )
    for i in range(len(YTrue)):
        cm[int(YTrue[i])][int(YPredict[i])] = cm[int(YTrue[i])][int(YPredict[i])] + 1
    true_values = np.sum(np.diagonal(cm))
    accuracy = float(true_values)/len(YTrue)
    return cm,accuracy