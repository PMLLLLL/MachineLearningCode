from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # ① 创建唯一的分类标签列表
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # ② 计算每种划分方式的信息熵
        infoGain = baseEntropy - newEntropy  # ③ 计算信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    # ① 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # ② 遍历完所有特征时，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优特征进行划分
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])

    # ③ 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
            return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')  # 使用二进制模式
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')  # 使用二进制模式
    return pickle.load(fr)

# myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# storeTree(myTree, 'classifierStorage.txt')
# loadedTree = grabTree('classifierStorage.txt')
# print(loadedTree)

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)

# myDat, labels = createDataSet()
# print(labels)
#
# myTree = treePlotter.retrieveTree(0)
# print(myTree)
#
# print(classify(myTree,labels,[1,0]))
# print(classify(myTree,labels,[1,1]))


# chooseBestFeatureToSplit(myDat)
# calcShannonEnt(myDat)

# >>> reload(trees)
# <module 'trees' from 'trees.pyc'>
# >>> myDat, labels = trees.createDataSet()
# >>> myTree = trees.createTree(myDat, labels)
# >>> myTree
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


# >>> reload(trees)
# <module 'trees' from 'trees.py'>
# >>> myDat, labels = trees.createDataSet()
# >>> trees.chooseBestFeatureToSplit(myDat)
# 0
# >>> myDat
# [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]


# >>> reload(trees.py)
# >>> myDat, labels = trees.createDataSet()
# >>> myDat
# [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# >>> trees.calcShannonEnt(myDat)
# 0.97095059445466858
