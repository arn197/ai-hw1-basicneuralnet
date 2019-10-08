from NeuralNet import *
from ConfusionMatrix import *
from Data import *
import math

def crossValidate(X_train, Y_train, X_test, Y_test, l_rate, epochs, reg_lambda, hidden_layer_nodes):
    model = train(X_train, Y_train, [l_rate, epochs, reg_lambda, hidden_layer_nodes, sigmoid_array, deltasig, X_train.shape[1], 1])
    # plotDecisionBoundary(model, X_train, Y_train)
    Y_pred = test(X_test,model)
    pf = getPerformanceScores(Y_test,Y_pred)
    print("Accuracy: {}\t Precision: {} \t Recall: {} \t F1: {}".format(pf["accuracy"],pf["precision"],pf["recall"],pf["f1"]))

def deltasig(x):
	return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_array(X):
	out = []
	for i in X:
		out.append(sigmoid(i))
	return out

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main():
	k = 5
	epochs = 10
	lr = 0.1
	reg_lambda = 1
	hidden_layer_nodes = 3
	X,Y = getData(["./Data/DataFor640/dataset1","lin"])
	K_split_data = splitData(X,Y,k)
	count = 1
	for itrain,itest in K_split_data:
		print("Split " + str(count))
		count += 1
		X_Train, Y_Train = X[itrain],Y[itrain]
		X_Test, Y_Test = X[itest],Y[itest]
		crossValidate(X_Train, Y_Train, X_Test, Y_Test, lr, epochs, reg_lambda, hidden_layer_nodes)

main()