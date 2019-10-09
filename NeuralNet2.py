import numpy as np
import os, sys

def deltasig(X):
	return sigmoid(X) * (1 - sigmoid(X))

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

class NeuralNetwork:
    def __init__(self, arch):
    	self.arch = arch
    	self.weights = [np.random.randn(self.arch[i], self.arch[i-1]) for i in range(1, len(self.arch))]
    	self.biases = [np.ones((n, 1)) for n in self.arch[1:]]

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
    	dl = self.delta_cross_entropy(y_pred, y)
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
            # print self.getCost(Y_pred, Y)
            for i in range(len(dw)):
            	self.weights[i] -= learningRate * dw[i]

    def predict(self, X):
        for i in xrange(len(self.weights)):
        	X = sigmoid(np.dot(self.weights[i], X))
        X[X > 0.5] = 1
        X[X <= 0.5] = 0
        print X
        return X

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def getCost(self, X, y):
        X = X.T.flatten()
        m = y.shape[0]
        p = self.stable_softmax(X)
        log_likelihood = - y * np.log(p[range(m)])
        loss = np.sum(log_likelihood) / m
        return loss

    def delta_cross_entropy(self, X, y):
        X = X.T.flatten()
        m = y.shape[0]
        grad = self.stable_softmax(X)
        grad[range(m)] -= y
        grad = grad/m
        return np.sum(grad)

def plotDecisionBoundary(model, X, Y):
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

def train(XTrain, YTrain, args):
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