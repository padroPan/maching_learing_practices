from numpy import *
import operator

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

group,labels = createDataset()
print(classify0([3,11],group,labels,3))
