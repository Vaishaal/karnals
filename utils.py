import csv
import numpy as np


ATCG_MAP = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
ATCG_REV_MAP = ['A', 'T', 'C', 'G']
ATCG = np.eye(4)

def convertSeqToMatrix(XSeq):
    X = np.zeros((len(XSeq), 4*len(XSeq[0])))
    for i,seq in enumerate(XSeq):
        x = np.zeros(4*100)
        X[i] = np.concatenate([ATCG[ATCG_MAP[x]] for x in seq])
    return X



def alphaVectorToString(alphavector, alpha_size=4):
    return ATCG_REV_MAP[np.argmax(alphavector)]

def vectorToString(vector, alpha_size=4):
    out = ''
    for i in range(vector.shape[0]/alpha_size):
        out += alphaVectorToString(vector[i*alpha_size:(i+1)*alpha_size])
    return out

def loadSeqFromText(seqFile, delim='\t'):
    XSeq = []
    labels = []
    with open(seqFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delim)
        for row in reader:
            seq = row['seq']
            label = row['Bound']
            XSeq.append(seq)
            labels.append(label)
    return XSeq[1:], np.array(labels[1:]).astype('uint8')


def loadFoldFromText(foldId):
    npzFile = foldId + ".npz"
    XTrainFile = foldId + "_Train.csv"
    XTestFile = foldId + "_Test.csv"
    yTrainFile = foldId + "_TrainLabels.csv"
    yTestFile = foldId + "_TestLabels.csv"
    return loadFromText(XTrainFile, XTestFile, yTrainFile, yTestFile)


def loadFromText(XTrainFile, XTestFile, yTrainFile, yTestFile):
    with open(XTrainFile) as XFile:
        print(("Loading XTrain... from " + XTrainFile))
        XTrainRaw = np.loadtxt(XFile, dtype="uint8", delimiter=",")

    with open(XTestFile) as XFile:
        print(("Loading XTest... from " + XTestFile))
        XTestRaw = np.loadtxt(XFile, dtype="uint8", delimiter=",")

    with open(yTrainFile) as yFile:
        print(("Loading yTrain... from " + yTrainFile))
        yTrainRaw = np.loadtxt(yFile, dtype="uint8", delimiter=",")
    with open(yTestFile) as yFile:
        print(("Loading yTest.. from " + yTestFile))
        yTestRaw = np.loadtxt(yFile, dtype="uint8", delimiter=",")
    return (XTrainRaw, XTestRaw, yTrainRaw, yTestRaw)

def loadFromNpz(npzFile):
    npz = np.load(npzFile)
    XTrainRaw = npz["XTrain"]
    XTestRaw = npz["XTest"]
    yTrainRaw = npz["yTrain"]
    yTestRaw = npz["yTest"]
    return XTrainRaw, XTestRaw, yTrainRaw, yTestRaw


def saveToNpz(fname, XTrain, XTest, yTrain, yTest):
    f = open(fname, "w+")
    numpy.savez(f, XTrain=XTrain, XTest=XTest, yTrain=yTrain, yTest=yTest)
    f.close()
    return 0

