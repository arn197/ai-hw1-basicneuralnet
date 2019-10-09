from NeuralNet2 import *
from ConfusionMatrix import *
from Data import *
import math

def crossValidate(X_train, Y_train, X_test, Y_test, l_rate, epochs, reg_lambda, architecture):
    model = train(X_train, Y_train, [l_rate, epochs, reg_lambda, architecture])
    # plotDecisionBoundary(model, X_train, Y_train)
    Y_pred = test(X_test,model)
    pf = getPerformanceScores(Y_test,Y_pred)
    print("Accuracy: {}\t Precision: {} \t Recall: {} \t F1: {}".format(pf["accuracy"],pf["precision"],pf["recall"],pf["f1"]))

def main():
	k = 5
	epochs = 10000
	lr = 0.1
	reg_lambda = 1
	architecture = [2, 10, 1]
	X,Y = getData(["./Data/DataFor640/dataset1","lin"])
	K_split_data = splitData(X,Y,k)
	count = 1
	for itrain,itest in K_split_data:
		print("Split " + str(count))
		count += 1
		X_Train, Y_Train = X[itrain],Y[itrain]
		X_Test, Y_Test = X[itest],Y[itest]
		crossValidate(X_Train, Y_Train, X_Test, Y_Test, lr, epochs, reg_lambda, architecture)

main()