"""
说明：
作者： fushichao
日期：2021年10月23日 15:57:10
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
输入：一个组点   dataSet = [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
     一组中心点 centroids = [[6, 4], [5, 4]]
'''
# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        # np.tile(data, (k, 1))将data沿x复制1倍  沿 y复制k倍
        # 将数据分成k组  计算每组与k个中心点的距离
        diff = np.tile(data, (k, 1)) - centroids
        squaredDiff = diff ** 2  # 平方
        # 按照行相加 即：对于一个点产生k个值，分别对应与k个中心点的距离 [25 34] k = 2时
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    # 返回一组 n行 k列的数据  n表示n个点  k表示k个中心
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

'''
clalist = 
[[0.         5.38516481]
 [1.         5.09901951]
 [1.         4.47213595]
 [5.83095189 1.        ]
 [5.38516481 0.        ]
 [5.         1.41421356]]

'''
'''
计算质心
'''

def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    # print(minDistIndices)  # [0 0 0 1 1 1]
    # DataFramte(dataSet)对DataSet分组，
    # groupby(min)按照min进行统计分类， 按minDistIndices对所有点进行分组
    # mean()对分类结果求均值
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean()
    newCentroids = newCentroids.values
    # 计算变化量
    changed = newCentroids - centroids
    return changed, newCentroids


'''
使用k-means分类

'''

def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序
    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)     # 调用欧拉距离 计算出n行 k列数据 n为点的数量 k为中心点数量
    minDistIndices = np.argmin(clalist, axis=1)  # 每行取最小值
    # print('------------')
    # print(minDistIndices)  # [0 0 0 1 1 1]
    for i in range(k):                           # 建立k个数组
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素 按照对每个点的分类分别添加到所属的类中
        cluster[j].append(dataSet[i])
    return centroids, cluster


if __name__ == '__main__':
    dataset = [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
    centroids, cluster = kmeans(dataset, 2)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i, j in cluster[0]:
        print()
        plt.scatter(i, j, marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
    for i, j in cluster[1]:
        plt.scatter(i, j, marker='o', color='blue', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()



