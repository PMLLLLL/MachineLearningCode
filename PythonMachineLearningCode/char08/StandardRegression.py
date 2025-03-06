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

# 标准回归计算
def standRegres(xArr,yArr):
    xMat = np.asmatrix(xArr); yMat = np.asmatrix(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 读取数据
xArr,yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr,yArr)

# 将数据转换成矩阵
xMat = np.asmatrix(xArr)
yMat = np.asmatrix(yArr)

print(ws)
# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(xMat[:,1].T.A, yMat.A,s=2,c='red')

xCopy = xMat.copy()
yHat = xCopy*ws #计算预测值
ax.plot(xCopy[:,1],yHat,linewidth=1)
plt.show()

# 计算相关系数
corr = np.corrcoef(yHat.T,yMat)
print(corr)
