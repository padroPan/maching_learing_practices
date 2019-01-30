from numpy import *
import operator
import os
import matplotlib
import matplotlib.pyplot as plt
#pip3 install tornado -i https://pypi.douban.com/simple/
#对 未知 类别 属性 的 数据 集中 的 每个 点 依次 执行 以下 操作：
# 1. 计算 已知 类别 数据 集 中的 点 与 当前 点 之间 的 距离；
# 2. 按照 距离 递增 次序 排序；
# 3. 选取 与 当前 点 距离 最小 的 k 个 点；
# 4. 确定 前 k 个 点 所在 类别 的 出现 频率；
# 5. 返回 前 k 个 点出 现 频率 最高 的 类别 作为 当前 点 的 预测 分类。

def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#参数： 用于 分类 的 输入 向量 是 inX，
# 输入 的 训练 样本 集 为 dataSet，
# 标签 向量 为 labels，
#  最后 的 参数 k 表示 用于 选择 最近 邻居 的 数目，
# 其中 标签 向量 的 元素 数目 和 矩阵 dataSet 的 行数 相同。
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #(以下三行)距离计算
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #(以下两行)选择距离最小的K个点
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] =classCount.get(voteIlabel,0) + 1
        #排序
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#group,labels = createDataset()
#print(classify0([3,11],group,labels,3))

def file2matrix(filename):
    fr = open(filename)
    array0lines=fr.readlines()
    #得到文件行数
    numberOfLines = len(array0lines)
    #创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in array0lines:
        #line.strip()截取掉所有的回车字符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat , classLabelVector
BASE_DIR=os.path.dirname(__file__)
#print(BASE_DIR)
FILE_PATH=os.path.join(BASE_DIR,'machinelearninginaction/','Ch02/','datingTestSet2.txt')
#print(FILE_PATH)
datingDataMat,datingLabels = file2matrix(FILE_PATH)
print(datingDataMat)
print(datingLabels)

fig = plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#normMat,ranges,minVals=autoNorm(datingDataMat)

#测试算法的方法
def datingClassTest():
    hoRatio = 0.1      #hold out 10%
    datingDataMat,datingLabels = file2matrix(FILE_PATH)       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

#datingClassTest()

def classfyPerson(videosP,ffMiles,iceCreamP):
    resultList=['not at all','in small doses','in large doses']
    persentTags=float(videosP)
    ffMiles=float(ffMiles)
    iceCream=float(iceCreamP)
    datingDataMat,datingLabels=file2matrix(FILE_PATH)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,persentTags,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult-1])

classfyPerson(30,50000,2)
