import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt

class TwoLayerMLP(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.
  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, arch, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['weights'] = []
    self.params['biases'] = []
    self.size = len(arch) - 1
    self.arch = arch
    for i in range(1, len(arch)):
        self.params['weights'].append(std * np.random.randn(arch[i - 1], arch[i]))
        self.params['biases'].append(np.zeros(arch[i]))
    # for i in self.params['weights']:
        # print i.shape
    # self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    # self.params['b1'] = np.zeros(hidden_size)
    # self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    # self.params['b2'] = np.zeros(output_size)
    # print self.params['W1'].shape
    # print self.params['W2'].shape

  def forward(self, X):
    z = X
    acts = []
    for i in range(self.size):
        W, b = self.params['weights'][i], self.params['biases'][i]
        # W2, b2 = self.params['W2'], self.params['b2']
        z = np.dot(z, W) + b  # 1st layer activation, N*H
        if i == self.size - 1:
            acts.append(z)
            break
        z = sp.expit(z)
        if i < self.size - 1:
            acts.append(z)
        # [PLEASE IMPLEMENT] 2nd layer activation, N*C
        # hint: involves W2, b2
        # scores = np.dot(h1,W2) + b2
    return acts

  def get_loss(self, X, y, scores, reg):
    N, D = X.shape
    A = np.max(scores, axis=1) # N*1
    F = np.exp(scores - A.reshape(N, 1))  # N*C
    P = F / np.sum(F, axis=1).reshape(N, 1)  # N*C
    loss = np.mean(-np.log(P[range(y.shape[0]), y]))
    # loss = np.mean(-np.choose(y, scores.T) + np.log(np.sum(F, axis=1)) + A)

    for i in range(self.size):
        # add regularization terms
        W = self.params['weights'][i]
        loss += 0.5 * reg * np.sum(W * W)
    return loss, P
  
  def backprop(self, X, y, P, acts, reg):
    W1, b1 = self.params['weights'][0], self.params['biases'][0]
    W2, b2 = self.params['weights'][1], self.params['biases'][1]
    _, C = W2.shape
    N, D = X.shape
    grads = {}
    ###########################################################################
    # write your own code where you see [PLEASE IMPLEMENT]
    #
    # Compute the backward pass, computing the derivatives of the weights
    # and biases. Store the results in the grads dictionary. For example,
    # grads['W1'] should store the gradient on W1, and be a matrix of same size
    ###########################################################################

    # output layer
    y_1hot = np.zeros((N,C))
    for i in range(N):
        y_1hot[i,y[i]] = 1
    dscore = P - y_1hot # [PLEASE IMPLEMENT] partial derivative of loss wrt. the logits (dL/dz)
    dW = [0] * self.size
    dB = [0] * self.size
    # dW2 = np.dot(h1.T, dscore)/N # partial derivative of loss wrt. W2
    # db2 = np.mean(dscore, axis=0)     # partial derivation of loss wrt. b2
    dW[-1] = np.dot(acts[-1].T, dscore)/N
    dB[-1] = np.mean(dscore, axis=0)
    acts = [X] + acts
    for i in range(self.size - 2, -1, -1):
        W = self.params['weights'][i + 1]
        dEy = np.dot(dscore, W.T)
        dscore = (acts[i + 1]*(1 - acts[i + 1])) * dEy
        dW[i] = np.dot(acts[i].T, dscore)/acts[i].shape[0]
        dB[i] = np.mean(dscore, axis = 0)
    # hidden layer
    # dhidden = np.dot(dscore,W2.T)
    # dz1 = (h1*(1-h1)) * dhidden

    # # first layer
    # dW1 = np.dot(X.T,dz1)/N # [PLEASE IMPLEMENT]
    # db1 = np.mean(dz1,axis=0) # [PLEASE IMPLEMENT]
    # ###########################################################################
    # #                            END OF YOUR CODE
    # ###########################################################################
    grads['weights'] = dW
    grads['biases'] = dB
    # grads['W2'] = dW2 + (reg * W2)
    # grads['b2'] = db2
    # grads['W1'] = dW1 + (reg * W1)
    # grads['b1'] = db1

    return grads


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary

    # Compute the forward pass
    ###########################################################################
    # write your own code where you see [PLEASE IMPLEMENT]
    #
    # Perform the forward pass, computing the class scores for the input.
    # Store the result in the scores variable, which should be an array of
    # shape (N, C).
    ###########################################################################
    scores, h1 = self.forward(X)

    ####################################s#######################################
    #                            END OF YOUR CODE
    ###########################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
      """
    scores_f = np.expit(scores)
    loss =
    """
    # cross-entropy loss with log-sum-exp
    loss, P = self.get_loss(X, y, scores, reg)
    grads = self.backprop(X, y, P, h1, reg)

    # Backward pass: compute gradients
    
    return loss, grads


  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_epochs=10, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    # num_train = X.shape[0]
    # iterations_per_epoch = max(num_train / batch_size, 1)
    # epoch_num = 0

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    grad_magnitude_history = []
    train_acc_history = []
    val_acc_history = []

    # np.random.seed(1)
    for epoch in range(num_epochs):
        # fixed permutation (within this epoch) of training data
        # perm = np.random.permutation(num_train)
        X1 = X
        Y1 = y
        # go through minibatches
        # for it in range(int(iterations_per_epoch)):
        # Create a random minibatch

        # Compute loss and gradients using the current minibatch
        acts = self.forward(X)
        loss, P = self.get_loss(X, y, acts[-1], reg)
        grads = self.backprop(X, y, P, acts[:-1], reg)
        # loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)

        # do gradient descent
        # for i in range(self.size):
        for i in range(self.size):
            self.params['weights'][i] -= grads['weights'][i] * learning_rate
            self.params['biases'][i] -= grads['biases'][i] * learning_rate
            # print self.params['weights'][1].shape, grads['weights'][1].shape
            # self.params['weights'] -= grads['weights'] * learning_rate
            # self.params['biases'] -= grads['biases'] * learning_rate
                # self.params[param] -= grads[param] * learning_rate

            # record gradient magnitude (Frobenius) for W1
            # grad_magnitude_history.append(np.linalg.norm(grads['W1']))

        # Every epoch, check train and val accuracy and decay learning rate.
        # Check accuracy
        # print getConfusionMatrix(y, self.predict(X))
        # print np.mean(self.predict(X) - y)
        train_acc = (self.predict(X1) == Y1).mean()
        # val_acc = (self.predict(X_va) == y_val).mean()
        train_acc_history.append(train_acc)
        # val_acc_history.append(val_acc)
        if verbose:
            print('Epoch %d: loss %f, train_acc %f'%(
                epoch+1, loss, train_acc))

        # Decay learning rate
        # learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'grad_magnitude_history': grad_magnitude_history, 
      'train_acc_history': train_acc_history,
      # 'val_acc_history': val_acc_history,
    }


  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    ###########################################################################
    # [PLEASE IMPLEMENT]
    # hint: it should be very easy
    y_pred = self.forward(X)[-1]
    y_pred =  np.argmax(np.exp(y_pred)/np.exp(np.sum(y_pred,axis=1)).reshape(-1,1),axis=1)
    # print y_pred
    ###########################################################################

    return y_pred

def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """
    len_labels = len(np.unique(YTrue))
    cm = np.zeros((len_labels ,len_labels ), int )
    for i in range(len(YTrue)):
        cm[int(YTrue[i])][int(YPredict[i])] = cm[int(YTrue[i])][int(YPredict[i])] + 1
    true_values = np.sum(np.diagonal(cm))
    accuracy = float(true_values)/len(YTrue)
    # print accuracy
    return cm,accuracy
                

def main():
    home = "/home/arn197/buf19/ai/p1/ai-hw1-basicneuralnet/"
    fx = open("Data/DataFor640/dataset1/LinearX.csv")
    fy = open("Data/DataFor640/dataset1/LinearY.csv")
    dfx = pd.read_csv(fx)
    dfy = pd.read_csv(fy)

    X = np.array(dfx)
    Y = np.array(dfy)
    Y_int = Y.astype(int)
    net = TwoLayerMLP([2, 3, 2], 2)
    stats = net.train(X, Y_int, learning_rate=0.00001, reg=1e-5, num_epochs=1000, verbose=True)
    # print('Final training loss: ', stats['loss_history'][-1])
    output = net.predict(X)
    print(getConfusionMatrix(Y_int,output.reshape(output.shape[0],1)))
    print('Final training loss: ', stats['loss_history'][-1])

    # plot the loss history and gradient magnitudes
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
          
if __name__ == '__main__':
    main()
