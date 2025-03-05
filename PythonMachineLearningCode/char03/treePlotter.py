import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import pause

# 指定支持中文的字体（如 SimHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 推荐
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 定义决策节点和叶子节点的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 决策节点样式
leafNode = dict(boxstyle="round4", fc="0.8")        # 叶节点样式
arrow_args = dict(arrowstyle="<-")                 # 箭头样式

# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]  # 获取第一个键（决策树的根节点）
    secondDict = myTree[firstStr]  # 获取根节点的子树
    for key in secondDict.keys():  # 遍历子树的所有分支
        if type(secondDict[key]).__name__ == 'dict':  # ① 测试节点是否为字典（判断是否是叶子节点）
            numLeafs += getNumLeafs(secondDict[key])  # 递归计算叶子节点数
        else:
            numLeafs += 1  # 叶子节点计数 +1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  # 获取第一个键（决策树的根节点）
    secondDict = myTree[firstStr]  # 获取根节点的子树
    for key in secondDict.keys():  # 遍历所有子节点
        if type(secondDict[key]).__name__ == 'dict':  # ① 测试节点是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key])  # 递归计算深度
        else:
            thisDepth = 1  # 叶子节点深度记为1
        if thisDepth > maxDepth:  # 记录最大深度
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]

# 绘制父节点与子节点之间的文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)

    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()





# myTree = retrieveTree(0)
# createPlot(myTree)
#
# myTree['no surfacing'][3] = 'maybe'
# createPlot(myTree)