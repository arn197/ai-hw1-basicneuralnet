import numpy as np
from itertools import permutations
#%%
def getData(dataDir):
	if dataDir[1]=="lin":
		X = np.loadtxt(dataDir[0]+"/LinearX.csv",delimiter=",")
		Y = np.loadtxt(dataDir[0]+"/LinearY.csv",delimiter=",")
	elif dataDir[1]=="nonlin":
		X = np.loadtxt(dataDir[0]+"/NonlinearX.csv",delimiter=",")
		Y = np.loadtxt(dataDir[0]+"/NonlinearY.csv",delimiter=",")
	else:
		X,Y=[],[]
		X.append(np.loadtxt(dataDir[0]+"/Digit_X_train.csv",delimiter=","))
		X.append(np.loadtxt(dataDir[0]+"/Digit_X_test.csv",delimiter=","))
		Y.append(np.loadtxt(dataDir[0]+"/Digit_y_train.csv",delimiter=","))
		Y.append(np.loadtxt(dataDir[0]+"/Digit_y_test.csv",delimiter=","))
		X=np.array(X)
		Y=np.array(Y)
	assert X.shape[0] == Y.shape[0]
	return X,Y

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
	assert X.shape[0] == Y.shape[0]

	m = X.shape[0]
	fold_size = m//K
	A = np.random.permutation(m).tolist()
	folds = list()
	for i in range(K):
		folds.append(A[i*fold_size:(i+1)*fold_size])
	K_Split = list()
	for i in range(len(folds)):
		l = list()
		for j in range(len(folds)):
			if i!=j:
				l = l+folds[j]
		K_Split.append([l,folds[i]])
	K_Split = np.array(K_Split)
	# print(K_Split[0][0][1])
	return K_Split
