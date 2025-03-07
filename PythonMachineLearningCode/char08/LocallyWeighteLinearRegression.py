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

def lwlr(testPoint, xarr, yarr, k=1.0):
    xmat = np.asmatrix(xarr); ymat = np.asmatrix(yarr).T
    m = np.shape(xmat)[0]
    weights = np.asmatrix(np.eye(m))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xmat[j, :]     #
        weights[j,j] = np.exp((diffMat*diffMat.T)[0,0]/(-2.0*k**2))
    xTx = xmat.T * (weights * xmat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xmat.T * (weights * ymat))
    return testPoint * ws

def lwlrTest(testArr, xarr, yarr, k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yhat = np.zeros(m)
    for i in range(m):
        yhat[i] = lwlr(testArr[i], xarr, yarr, k).item()
    return yhat

# 读取数据
xArr,yArr = loadDataSet('ex0.txt')
# 将数据转换成矩阵
xMat = np.asmatrix(xArr)
yMat = np.asmatrix(yArr)

# 设定要使用的k值
kList = [1,0.01,0.003]

fig = plt.figure()

for K in kList:
    # 计算估计值
    yHat = lwlrTest(xArr,xArr, yArr,K)

    # 排序
    srtInd = xMat[:,1].argsort(0)

    # 绘图
    ax = fig.add_subplot(len(kList),1,kList.index(K)+1)
    ax.scatter(xMat[:,1].T.A, yMat.A,s=2,c='red')
    ax.plot(xMat[srtInd,1],yHat[srtInd],linewidth=1)
    ax.set_title('k = %.3f' % K)

# 调整子图间距
plt.subplots_adjust(hspace=0.5)
plt.show()

