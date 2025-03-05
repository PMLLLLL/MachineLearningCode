import chardet
from numpy import *
import math

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
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

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

import numpy as np

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = np.ones(numWords);p1Num = np.ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)  # change to log()
    p0Vect = np.log(p0Num / p0Denom)  # change to log()

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# def testingNB():
# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# trainMat = []
# for postinDoc in listOposts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
# testEntry = ['love', 'my', 'dallmation']
# thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
# print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
# testEntry = ['stupid', 'garbage']
# thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
# print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

import random
#def spamTest():
docList = []; classList = []; fullText = []

# 检测文件编码格式

for i in range(1, 26):
    with open('email/spam/%d.txt' % i, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    wordList = textParse(open('email/spam/%d.txt' % i,encoding=encoding).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(1)

    with open('email/ham/%d.txt' % i, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    wordList = textParse(open('email/ham/%d.txt' % i,encoding=encoding).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(0)
vocabList = createVocabList(docList)

# 随机删除10个数据作为验证集，剩下40个作为训练集
trainingset = list(range(50)); testSet = []
for i in range(10):
    randIndex = int(random.uniform(0, len(trainingset)))
    testSet.append(trainingset[randIndex])
    del(trainingset[randIndex])
trainMat = []; trainClasses = []
for docIndex in trainingset:
    trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
    trainClasses.append(classList[docIndex])
pOV, p1V, pspam = trainNB0(np.array(trainMat), np.array(trainClasses))
errorCount = 0
for docIndex in testSet:
    wordVector = setOfWords2Vec(vocabList, docList[docIndex])
    if classifyNB(np.array(wordVector), pOV, p1V, pspam) != classList[docIndex]:
        errorCount += 1
        print(docList[docIndex],classList[docIndex])
print('the error rate is: ', float(errorCount) / len(testSet))


# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# print(myVocabList)
#
# trainMat = []
# for postinDoc in listOposts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
# print(trainMat)
# print(len(trainMat))
#
#
# p0V, p1V, pAb=trainNB0(trainMat,listClasses)
# print(pAb)
# print(p0V)
# print(p1V)

