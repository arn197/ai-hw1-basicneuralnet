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
        self.W = np.random.rand(n_connections, n_nodes)
        self.nodes = [Node()] * n_nodes

    def forward_update(self, X, Y):
        for i in range(self.n_nodes):
            self.nodes[i].forward_update(X[i], Y[i])

    def back_update(self, W):
        self.W = W

class NeuralNetwork:
    def __init__(self, NNodes, activate, deltaActivate, inputDimension, n_classes):
        self.NNodes = NNodes # the number of nodes in the hidden layer
        self.activate = activate # a function used to activate
        self.deltaActivate = deltaActivate # the derivative of activate
        self.inputDimension = inputDimension
        self.input_layer = Layer(NNodes, inputDimension)
        self.hidden_layer = Layer(1, NNodes)
        self.output_layer = Layer(1, n_classes)
        self.bias = 1
    
    def fit(self, X, Y, learningRate, epochs, regLambda):
        for epoch in xrange(epochs):
            for i in xrange(len(X)):
                x = X[i]
                y = Y[i]
                self.forward(x)
                l, dl = self.getCost(self.output_layer.nodes, y)
                self.backpropagate(l, dl, learningRate)

    def predict(self, X):
        y = []
        for i in xrange(len(X)):
            self.forward(X[i])
            y.append(self.output_layer.nodes[0].y)
        return np.array(y)

    def forward(self, X):
        # X = np.append(X, self.bias)
        X = np.dot(self.input_layer.W, X)
        self.hidden_layer.forward_update(X, self.activate(X))
        X = self.activate(X)
        # X = np.append(X, self.bias)
        X = np.dot(self.hidden_layer.W, X)
        self.output_layer.forward_update(X, self.activate(X))
        X = self.activate(X)

    def updateWeights(self, W, nodes, forward_nodes, layer, lr):
        for i in xrange(len(nodes)):
            dlx = 0
            node = nodes[i]
            for j in xrange(len(forward_nodes)):
                fnode = forward_nodes[j]
                dly = fnode.dlx * W[j][i]
                W[j][i] -= lr * node.y * fnode.dlx
                dlx += dly
            dlx *= self.deltaActivate(node.x)
            node.back_update(dlx)
        layer.back_update(W)
        
    def backpropagate(self, l, dl, lr):
        # Compute gradient for each layer.
        #Output layer update
        for i in xrange(len(self.output_layer.nodes)):
            output_node = self.output_layer.nodes[i]
            dlx = dl * self.deltaActivate(output_node.x)
            output_node.back_update(dlx)

        W = self.hidden_layer.W
        nodes = self.hidden_layer.nodes
        forward_nodes = self.output_layer.nodes
        self.updateWeights(W, nodes, forward_nodes, self.hidden_layer, lr)

        W = self.input_layer.W
        nodes = self.input_layer.nodes
        forward_nodes = self.hidden_layer.nodes
        self.updateWeights(W, nodes, forward_nodes, self.input_layer, lr)

        # Update weight matrices.
        
    def getCost(self, YPredict, YTrue):
        cost = 0
        cost += (YPredict[0].y - YTrue)**2
        return cost, -1
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
    lr = args[0]
    epochs = args[1]
    reg = args[2]
    hidden_layer = args[3]
    sigmoid = args[4]
    deltasig = args[5]
    inputDimension = args[6]
    n_classes = args[7]

    model = NeuralNetwork(hidden_layer, sigmoid, deltasig, inputDimension, n_classes)
    model.fit(XTrain, YTrain, lr, epochs, reg)
    return model
    
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
    return model.predict(XTest)

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