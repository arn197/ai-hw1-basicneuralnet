import numpy as np
from itertools import permutations
#%%
def getData(dataDir):
    if dataDir[1]=="lin":
        X = np.loadtxt(dataDir[0]+"/LinearX.csv",delimiter=",")
        Y = np.loadtxt(dataDir[0]+"/LinearY.csv",delimiter=",")
    else:
        X = np.loadtxt(dataDir[0]+"/NonlinearX.csv",delimiter=",")
        Y = np.loadtxt(dataDir[0]+"/NonlinearY.csv",delimiter=",")
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

def crossValidate(X_train,Y_train,X_test,Y_test,l_rate,epochs, reg_lambda):
    model = train(X_train,Y_train, [l_rate,epochs,reg_lambda])
    plotDecisionBoundary(model, X_train, Y_train)
    Y_pred = test(X_test,model)
    pf = getPerformanceScores(Y_test,Y_pred)
    print("Accuracy: {}\t Precision: {} \t Recall: {} \t F1: {}".format(pf["accuracy"],pf["precision"],pf["recall"],pf["f1"]))



#%%
# X,Y = getData(["./Data/DataFor640/dataset1","lin"])
# # print(X.shape)
# # print(Y.shape)
# Ksp = splitData(X,Y)
# print(Ksp[0])
#%%
def main():
    X,Y = getData(["./Data/DataFor640/dataset1","lin"])
    K_split_data = splitData(X,Y)

    for itrain,itest in K_split_data:
        
        X_Train, Y_Train = X[itrain],Y[itrain]
        X_Test, Y_Test = X[itest],Y[itest]
        print(X_train[:5])

        crossValidate(X_Train,Y_Train,X_Test,Y_Test,lr,ep,reg_lambda)
main()
