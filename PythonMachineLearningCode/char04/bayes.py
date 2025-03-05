from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个全零向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 词存在时，置1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    return returnVec

import numpy as np

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num / p1Denom  # change to log()
    p0Vect = p0Num / p0Denom  # change to log()

    return p0Vect, p1Vect, pAbusive


listOposts, listClasses = loadDataSet()
myVocabList = createVocabList(listOposts)
print(myVocabList)

trainMat = []
for postinDoc in listOposts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
print(trainMat)
print(len(trainMat))


p0V, p1V, pAb=trainNB0(trainMat,listClasses)
print(pAb)
print(p0V)
print(p1V)

