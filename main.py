import numpy as np
import os, sys

class Node:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dlx = 0

    def forward_update(self, x, y):
        self.x = x
        self.y = y

    def back_update(self, dlx):
        self.dlx = dlx

class Layer:
    def __init__(self, n_connections, n_nodes):
        self.n_nodes = n_nodes
        self.W = np.zeros((n_connections, n_nodes))
        self.nodes = np.zeros((n_nodes, 1))
        for i in xrange(n_nodes):
            self.nodes[i][0] = Node()

    def forward_update(self, X, Y):
        for i in xrange(self.n_nodes):
            self.nodes[i][0].forward_update(X[i][0], Y[i][0])

    def back_update(self, W):
        self.W = W

class NeuralNetwork:
    def __init__(self, NNodes, activate, deltaActivate, inputDimension, loss, n_classes):
        self.NNodes = NNodes # the number of nodes in the hidden layer
        self.activate = activate # a function used to activate
        self.deltaActivate = deltaActivate # the derivative of activate
        self.inputDimension = inputDimension
        self.input_layer = Layer(NNodes, inputDimension + 1)
        self.hidden_layer = Layer(1, NNodes + 1)
        self.output_layer = Layer(1, n_classes)
        self.loss = loss
    
    def fit(self, X, Y, learningRate, epochs, regLambda):
        for epoch in xrange(epochs):
            for i in xrange(len(X)):
                x = X[i]
                y = Y[i]
                forward(x)
                l, dl = getCost(output_layer.nodes, y)
                backpropagate(l, dl, learningRate)

    def predict(self, X):
        for i in xrange(len(X)):
            forward(X)
        y = numpy.zeros((len(X), 1))
        for i in xrange(len(X)):
            y[i] = self.output_layer.nodes[i][0].y
        return y

    def forward(self, X):
        X = np.dot(input_layer.W, X)
        hidden_layer.forward_update(X, activation(X))
        X = activation(X)
        X = np.dot(hidden_layer.W, X)
        output_layer.forward_update(X, activation(X))
        X = activation(X)

    def updateWeights(self, W, nodes, forward_nodes, layer):
        for i in xrange(len(nodes)):
            dlx = 0
            node = nodes[i][0]
            for j in xrange(len(forward_nodes)):
                fnode = forward_nodes[j][0]
                dly = fnode.dlx * W[i][j]
                W[i][j] -= lr * node.y * fnode.dlx
                dlx += dly
            dlx *= self.deltaActivate(node.x)
            node.back_update(dlx)
        layer.back_update(W)
        
    def backpropagate(self, l, dl, lr):
        # Compute gradient for each layer.
        #Output layer update
        for i in xrange(len(output_layer.nodes)):
            output_node = output_layer.nodes[i][0]
            dlx = dl * self.deltaActivate(output_node.x)
            output_node.back_update(dlx)

        W = hidden_layer.W
        nodes = hidden_layer.nodes
        forward_nodes = output_layer.nodes
        updateWeights(W, nodes, forward_nodes, hidden_layer)

        W = input_layer.W
        nodes = input_layer.nodes
        forward_nodes = hidden_layer.nodes
        updateWeights(W, nodes, forward_nodes, input_layer)

        # Update weight matrices.
        
    def getCost(self, YTrue, YPredict):
        return np.sum((YPredict - YTrue)**2) / len(YTrue), np.sum(YTrue - YPredict)
        # Compute loss / cost in terms of crossentropy.
        # (hint: your regularization term should appear here)

def getData(dataDir):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''
    # TO-DO for this part:
    # Use your preferred method to read the csv files.
    # Write your codes here:
    
    
    # Hint: use print(X.shape) to check if your results are valid.
    return X, Y

def splitData(X, Y, K = 5):
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    '''
    
    # Make sure you shuffle each train list.
    pass

def plotDecisionBoundary(model, X, Y):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

def train(XTrain, YTrain, args):
    """
    This function is used for the training phase.
    Parameters
    ----------
    XTrain : numpy matrix
        The matrix containing samples features (not indices) for training.
    YTrain : numpy array
        The array containing labels for training.
    args : List
        The list of parameters to set up the NN model.
    Returns
    -------
    NN : NeuralNetwork object
        This should be the trained NN object.
    """
    # 1. Initializes a network object with given args.
    
    
    # 2. Train the model with the function "fit".
    # (hint: use the plotDecisionBoundary function to visualize after training)
    
    
    # 3. Return the model.
    
    pass

def test(XTest, model):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """
    pass

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
    pass
    
def getPerformanceScores(YTrue, YPredict):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    pass