import numpy as np
import os, sys

def deltasig(X):
	return sigmoid(X) * (1 - sigmoid(X))

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

class NeuralNetwork:
    def __init__(self, arch):
    	# self.weights = 
    	self.arch = arch
    	self.weights = [np.random.randn(self.arch[i], self.arch[i-1]) for i in range(1, len(self.arch))]
    	self.biases = [np.random.randn(n, 1) for n in self.arch[1:]]

    def forward(self, X):
    	pre_act = []
    	act = []
    	for i in range(0, len(self.weights)):
    		W = self.weights[i]
    		B = self.biases[i]
    		X = np.dot(W, X) + B
    		pre_act.append(X)
    		X = sigmoid(X)
    		act.append(X)
    	return X, act, pre_act

    def mse(self, y_pred, y):
    	return (1/2.0) * float(np.sum((y_pred - y) ** 2))/len(y)

    def mse_prime(self, y_pred, y):
    	return y_pred - y

    def get_gradients(self, pre_act, y_pred, y):
    	dl = self.mse_prime(y_pred, y)
    	dw = [0] * (len(self.arch))
    	dw[-1] = dl * deltasig(pre_act[-1])
    	# print self.weights[0].shape
    	pre_act = [0] + pre_act
    	for i in range(len(dw) - 2, -1, -1):
    		dw[i] = np.dot(self.weights[i].T, dw[i + 1]) * deltasig(pre_act[i])
    	return dw

    def backpropagate(self, gradients, pre_act, act):
    	dw = []
    	db = []
    	for i in range(len(self.arch) - 1):
    		dw.append(np.dot(act[i], gradients[i].T))
    		db.append(np.expand_dims(gradients[i].mean(axis=1), 1))
    	return dw, db

    def fit(self, X, Y, learningRate, epochs, regLambda):
        X = X.T
        for epoch in xrange(epochs):
            Y_pred, act, pre_act = self.forward(X)
            gradients = self.get_gradients(pre_act, Y_pred, Y)
            dw, db = self.backpropagate(gradients, pre_act, act)
            # print self.weights
            for i in range(len(dw)):
            	# print dw[i].shape, self.weights[i].shape
            	# print db[i].shape
            	# print self.biases[i].shape, db[i].shape
            	self.weights[i] -= learningRate * dw[i]
            # print self.predict(X)
            	# self.biases[i] -= learningRate * db[i]
            # l = self.getCost(Y_pred, Y)
            # dl = self.delta_cross_entropy(Y_pred, Y)
            # self.backpropagate(l, dl, learningRate)
            # for i in xrange(len(X)):
            #     x = X[i]
            #     y = Y[i]
            #     Y_pred = self.forward(x)
            #     l, dl = self.getCost(self.output_layer.nodes, y)
            #     self.backpropagate(l, dl, learningRate)

    def predict(self, X):
        for i in xrange(len(self.weights)):
        	X = sigmoid(np.dot(self.weights[i], X) + self.biases[i])
        pred = (X > 0.7).astype(int)
        return pred
        # y = []
        # for i in xrange(len(X)):
        #     self.forward(X[i])
        #     y.append(self.output_layer.nodes[0].y)
        # return np.array(y)

    # def forward(self, X):
    #     # X = np.append(X, self.bias)
    #     self.input_layer.forward_update(X, X)

    #     X = np.dot(self.input_layer.W, X)
    #     self.hidden_layer.forward_update(X, self.activate(X))
    #     X = self.activate(X)
    #     # X = np.append(X, self.bias)
    #     X = np.dot(self.hidden_layer.W, X)
    #     self.output_layer.forward_update(X, self.activate(X))
    #     X = self.activate(X)
    #     return X

    # def updateWeights(self, W, nodes, forward_nodes, layer, lr):
    #     for i in xrange(len(nodes)):
    #         node = nodes[i]
    #         # print node.y
    #         # print node.x
    #         m = node.y.shape[0]
    #         dlx = np.zeros((m))
    #         for j in xrange(len(forward_nodes)):
    #             fnode = forward_nodes[j]
    #             dly = fnode.dlx * W[j][i]
    #             W[j][i] -= lr * np.dot(node.y,fnode.dlx)
    #             dlx += dly
    #         dlx *= self.deltaActivate(node.x)
    #         node.back_update(dlx)
    #     layer.back_update(W)
        
    # def backpropagate(self, l, dl, lr):
    #     # Compute gradient for each layer.
    #     #Output layer update
    #     for i in xrange(len(self.output_layer.nodes)):
    #         output_node = self.output_layer.nodes[i]
    #         dlx = dl * self.deltaActivate(output_node.x)
    #         output_node.back_update(dlx)

    #     W = self.hidden_layer.W
    #     nodes = self.hidden_layer.nodes
    #     forward_nodes = self.output_layer.nodes
    #     self.updateWeights(W, nodes, forward_nodes, self.hidden_layer, lr)

    #     W = self.input_layer.W
    #     nodes = self.input_layer.nodes
    #     forward_nodes = self.hidden_layer.nodes
    #     self.updateWeights(W, nodes, forward_nodes, self.input_layer, lr)

        # Update weight matrices.

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def getCost(self, X, y):
        X = X.T.flatten()
        m = y.shape[0]
        p = X
        log_likelihood = - y * np.log(p[range(m)])
        loss = np.sum(log_likelihood) / m
        return loss

    def delta_cross_entropy(self, X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        X = X.T.flatten()
        m = y.shape[0]
        grad = self.stable_softmax(X)
        grad[range(m)] -= y
        grad = grad/m
        return np.sum(grad)

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
    arch = args[3]

    model = NeuralNetwork(arch)
    model.fit(XTrain, YTrain, lr, epochs, reg)
    return model

def test(XTest, model):
    out = model.predict(XTest.T)
    return out