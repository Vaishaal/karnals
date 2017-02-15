import scipy.linalg
import sklearn.metrics as metrics
import numpy as np


def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    return y[:, 1]


def learnPrimal(trainData, labels, W=None, reg=0.1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData, dtype=np.float32).reshape(trainData.shape[0], -1)
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
    return (train_roc, test_roc, train_pr_auc, test_pr_auc)

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
    train_pr_auc = metrics.average_precision_score(labelsTrain, yTrainHat)
    test_pr_auc = metrics.average_precision_score(labelsTest, yTestHat)
    return (train_roc, test_roc, train_pr_auc, test_pr_auc)

