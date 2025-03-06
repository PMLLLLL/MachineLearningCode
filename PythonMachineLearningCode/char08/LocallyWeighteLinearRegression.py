import numpy as np
import matplotlib.pyplot as plt

# 载入数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.asmatrix(xArr); yMat = np.asmatrix(yArr).T
    m = np.shape(xMat)[0]
    weights = np.asmatrix(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
# 读取数据
xArr,yArr = loadDataSet('ex0.txt')
# 将数据转换成矩阵
xMat = np.asmatrix(xArr)
yMat = np.asmatrix(yArr)

# 计算估计值
yHat = lwlrTest(xArr,xArr, yArr,0.003)
# 绘图

# 排序
srtInd = xMat[:,1].argsort(0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].T.A, yMat.A,s=2,c='red')
ax.plot(xMat[srtInd,1],yHat[srtInd],linewidth=1)
plt.show()

