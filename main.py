from NeuralNet_ML import *
from Data import *
import random



def crossValidate(X_train, Y_train, X_test, Y_test, l_rate, epochs, reg_lambda, architecture):
	model = train(X_train, Y_train, [l_rate, epochs, reg_lambda, architecture])
	# plotDecisionBoundary(model, X_train, Y_train)
	Y_pred = test(X_test,model)
	pf = getPerformanceScores(Y_test,Y_pred)
	print("Accuracy: {}\t Precision: {} \t Recall: {} \t F1: {}".format(pf["accuracy"],pf["precision"],pf["recall"],pf["f1"]))

# def main():
# 	k = 5
# 	epochs = 10000
# 	lr = 0.1
# 	reg_lambda = 1
# 	architecture = [2, 10, 1]
# 	X,Y = getData(["./Data/DataFor640/dataset1","lin"])
# 	K_split_data = splitData(X,Y,k)
# 	count = 1
# 	for itrain,itest in K_split_data:
# 		print("Split " + str(count))
# 		count += 1
# 		X_Train, Y_Train = X[itrain],Y[itrain]
# 		X_Test, Y_Test = X[itest],Y[itest]
# 		crossValidate(X_Train, Y_Train, X_Test, Y_Test, lr, epochs, reg_lambda, architecture)

def main():
	X,Y = getData(["./Data/DataFor640/dataset1","lin"])

	# Y_int = Y.astype(int)
	  #     X.reshape(897*64,1)
	K_split_data = splitData(X,Y)
	count = 1
	for itrain,itest in K_split_data:
		print("Split " + str(count))
		count += 1
		X_Train, Y_Train = X[itrain],Y[itrain].astype(int)
		X_Test, Y_Test = X[itest],Y[itest].astype(int)

		net = TwoLayerMLP(X_Train.shape[1],3,2,5,"sigmoid")
		stats = net.train(X_Train, Y_Train, X_Train, Y_Train,learning_rate=1.2, reg=1e-1,  num_epochs=100, verbose=True)
		output = net.predict(X_Test)
		print(getConfusionMatrix(Y_Test,output))
		print('Final training loss: ', stats['loss_history'][-1])

		# plot the loss history and gradient magnitudes
		plt.subplot(2, 1, 1)
		plt.plot(stats['loss_history'])
		plt.xlabel('epoch')
		plt.ylabel('training loss')
		plt.title('Training Loss history')
		plt.show()



if __name__ == '__main__':
	random.seed(0)
	main()
