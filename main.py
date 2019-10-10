from NeuralNet_ML import *
from Data import *
from ConfusionMatrix import *
import random
import pandas as pd

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
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
    plt.show()


def crossValidate(X_train, Y_train, X_test, Y_test, l_rate, epochs, reg_lambda, params, plot=False):

	net = TwoLayerMLP(params[0],params[1],params[2],params[3],"sigmoid")
	stats = net.train(X_train, Y_train,learning_rate=l_rate, reg=reg_lambda,  num_epochs=epochs, verbose=False)
	output = net.predict(X_test)
	if plot:
		plotDecisionBoundary(net, X_train, Y_train)
	print('Final training loss: ', stats['loss_history'][-1])
	print("-----Confusion Matrix-----")
	scores=getPerformanceScores(Y_test,output)
	print("Overall Accuracy: ",scores["accuracy"])
	for i in range(scores['CM'].shape[0]):
		print("--Label ",i,"--")
		print("Precision: ",scores["precision"][i])
		print("Recall: ",scores["recall"][i])
		print("F1-score: ",scores["f1"][i])
	return stats

def main():

	print("-------Dataset 1:Linear-------")
	X,Y = getData(["./Data/DataFor640/dataset1","lin"])

	# Y_int = Y.astype(int)
	  #     X.reshape(897*64,1)
	K_split_data = splitData(X,Y)
	count = 1
	for itrain,itest in K_split_data:
		print("Split " + str(count))
		X_Train, Y_Train = X[itrain],Y[itrain].astype(int)
		X_Test, Y_Test = X[itest],Y[itest].astype(int)

		stats = crossValidate(X_Train, Y_Train, X_Test, Y_Test, 1.2, 200, 1e-2, [X_Train.shape[1],3,2,5], True)

		# plot the loss history and gradient magnitudes
		plt.subplot(2, 1, 1)
		plt.plot(stats['loss_history'])
		plt.xlabel('epoch')
		plt.ylabel('training loss')
		plt.title('Linear Training Loss history - Split '+str(count))
		plt.show()

		count += 1

	print("-------Dataset 1:Non Linear-------")
	X,Y = getData(["./Data/DataFor640/dataset1","nonlin"])

	# Y_int = Y.astype(int)
	  #     X.reshape(897*64,1)
	K_split_data = splitData(X,Y)
	count = 1
	for itrain,itest in K_split_data:
		print("Split " + str(count))
		X_Train, Y_Train = X[itrain],Y[itrain].astype(int)
		X_Test, Y_Test = X[itest],Y[itest].astype(int)

		stats = crossValidate(X_Train, Y_Train, X_Test, Y_Test, 2, 300, 1e-3, [X_Train.shape[1],3,2,5], True)

		# plot the loss history and gradient magnitudes
		plt.subplot(2, 1, 1)
		plt.plot(stats['loss_history'])
		plt.xlabel('epoch')
		plt.ylabel('training loss')
		plt.title('Non Linear Training Loss history - Split '+str(count))
		plt.show()

		count += 1






	print("-------Dataset 2:Digits-------")
	X,Y = getData(["./Data/DataFor640/dataset2","digits"])
	X_Train,X_Test = X[0],X[1]
	Y_Train,Y_Test = Y[0].astype(int),Y[1].astype(int)
	# net = TwoLayerMLP(X_Train.shape[1],10,150,2,"sigmoid")
	stats = crossValidate(X_Train, Y_Train, X_Test, Y_Test, 1.5 , 250, 1e-2, [X_Train.shape[1],10,150,2])


	# plot the loss history and gradient magnitudes
	plt.subplot(2, 1, 1)
	plt.plot(stats['loss_history'])
	plt.xlabel('epoch')
	plt.ylabel('training loss')
	plt.title('Training Loss history')





if __name__ == '__main__':
	# random.seed(0)
	main()
