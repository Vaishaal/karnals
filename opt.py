import scipy.linalg
import sklearn.metrics as metrics
import numpy as np
import numba
from numba import jit


def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    return y[:, 1]


def learnPrimal(trainData, labels, W=None, reg=0.1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData, dtype=np.float32).reshape(trainData.shape[0], -1).copy()
    if (W == None):
        W = np.ones(n)[:, np.newaxis]

    print("X SHAPE ", trainData.shape)
    print("Computing XTX")
    sqrtW = np.sqrt(W)
    X *= sqrtW
    XTWX = X.T.dot(X)
    print("Done Computing XTX")
    idxes = np.diag_indices(XTWX.shape[0])
    XTWX[idxes] += reg
    y = np.eye(max(labels) + 1)[labels]
    XTWy = X.T.dot(W * y)
    model = scipy.linalg.solve(XTWX, XTWy)
    return model

def trainAndEvaluateDualModel(KTrain, KTest, labelsTrain, labelsTest, reg=0.1):
    model = learnDual(KTrain,labelsTrain, reg=reg)
    predTrainWeights = evaluateDualModel(KTrain, model)
    predTestWeights = evaluateDualModel(KTest, model)
    train_roc = metrics.roc_curve(labelsTrain, predTrainWeights)
    test_roc = metrics.roc_curve(labelsTest, predTestWeights)
    train_pr_auc = metrics.average_precision_score(labelsTrain, predTrainWeights)
    test_pr_auc = metrics.average_precision_score(labelsTest, predTestWeights)
    return (train_roc, test_roc, train_pr_auc, test_pr_auc, predTrainWeights, predTestWeights)

def learnDual(gramMatrix, labels, reg=0.1, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model")
    y = np.eye(max(labels) + 1)[labels]
    idxes = np.diag_indices(gramMatrix.shape[0])
    gramMatrix /= float(TOT_FEAT)
    print("reg is " + str(reg))
    gramMatrix[idxes] += (reg)
    model = scipy.linalg.solve(gramMatrix, y)
    gramMatrix[idxes] -= (reg)
    gramMatrix *= TOT_FEAT
    return model


def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.0, W=None):
    print(XTrain.shape)
    print(XTest.shape)
    model = learnPrimal(XTrain, labelsTrain, reg=reg, W=W)
    yTrainHat = XTrain.dot(model)[:,1]
    yTestHat = XTest.dot(model)[:,1]

    yTrainPred = np.argmax(XTrain.dot(model), axis=1)
    yTestPred = np.argmax(XTest.dot(model), axis=1)

    print("Train acc", metrics.accuracy_score(yTrainPred, labelsTrain))
    print("Test acc", metrics.accuracy_score(yTestPred, labelsTest))

    train_roc = metrics.roc_curve(labelsTrain, yTrainHat)
    test_roc = metrics.roc_curve(labelsTest, yTestHat)
    train_pr = metrics.precision_recall_curve(labelsTrain, yTrainHat)
    test_pr = metrics.precision_recall_curve(labelsTest, yTestHat)
    train_pr_auc = metrics.average_precision_score(labelsTrain, yTrainHat)
    test_pr_auc = metrics.average_precision_score(labelsTest, yTestHat)
    return (train_roc, test_roc, train_pr, test_pr, train_pr_auc, test_pr_auc, yTrainHat, yTestHat)



@jit(nopython=True)
def rbf(K, gamma):
    for x in range(K.shape[0]):
        for y in range(K.shape[1]):
            K[x,y] = np.exp(gamma*K[x,y])
    return K


def computeRBFGramMatrix(K, XTrainNorms, XTestNorms, gamma=1, gamma_sample=10000):
    XTrainNorms = XTrainNorms.reshape(XTrainNorms.shape[0], 1)
    XTestNorms = XTestNorms.reshape(XTestNorms.shape[0], 1)
    print("TURNING K -> DISTANCE")
    K *= -2
    K += XTrainNorms.T
    K += XTestNorms 
    if (gamma == None):
        print("Calculating gamma")
        samples = numpy.random.choice(K.shape[0], gamma_sample*2, replace=False)
        x1 = samples[:gamma_sample]
        x2 = samples[gamma_sample:]
        sample_d = K[x1, x2]
        print("Sample d shape ", sample_d.shape)
        median = numpy.median(sample_d)
        gamma = 2.0/median
        print(gamma)
    gamma = -1.0 * gamma

    print(np.max(K))
    print(np.min(K))
    print("Computing RBF")
    return rbf(K, gamma), -1.0*gamma

def computeDistanceMatrix(XTest, XTrain):
    XTrain = XTrain.reshape(XTrain.shape[0], -1)
    XTest = XTest.reshape(XTest.shape[0], -1)
    XTrain_norms = (np.linalg.norm(XTrain, axis=1) ** 2)[:, np.newaxis]
    XTest_norms = (np.linalg.norm(XTest, axis=1) ** 2)[:, np.newaxis]
    K = XTest.dot(XTrain.T)
    K *= -2
    K += XTrain_norms.T
    K += XTest_norms  
    return K
