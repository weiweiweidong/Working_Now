# -*- coding: utf-8 -*-
########################## 这部分是为了设置相对路径而作的改动 ##########################
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前脚本所在目录
os.chdir(script_dir)
# 打印当前工作目录以确认
print("Current working directory:", os.getcwd())
##################################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def main():
    # 导入数据
    data = np.loadtxt("discord.txt", delimiter="\t")
    # 第100～2000条数据为训练集，2001～6000为测试集
    train_data = data[100:2000, 2]
    test_data = data[2001:6000, 2]

    # 打印一下原始数据
    plot_data(data)

    # train_data = moving_average(train_data, 20)
    # test_data = moving_average(test_data, 20)

    width = 5
    nk = 1

    # 时间序列嵌入
    """
    这里的embed实现的功能是：
        如果向量为 [1, 2, 3, 4, 5, 6, 7, 8, 9]，dim=3
        那么 embedded = embed(lst, dim) 的结果是：
        array([[3., 2., 1.],
                [4., 3., 2.],
                [5., 4., 3.],
                [6., 5., 4.],
                [7., 6., 5.],
                [8., 7., 6.],
                [9., 8., 7.]])
    """
    train = embed(train_data, width)
    test = embed(test_data, width)
    neigh = NearestNeighbors(
        n_neighbors=nk
    )  # 创建一个最近邻对象，查询点只寻找一个最近邻居
    neigh.fit(train)  # 使用train集来 拟合数据
    d, ind = neigh.kneighbors(
        test
    )  # 使用test集合，返回test中每个点的最近邻的距离 和 索引
    d = np.mean(d, axis=1)  # 求平均值
    mx = np.max(d)  # 获取最大值
    d = d / mx  # 求百分比

    # プロット
    test_for_plot = data[2001 + width : 6000, 2]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    # p1, = ax1.plot(d, '-m',linewidth = 1, linestyle="dotted" )
    (p1,) = ax1.plot(d, "-b", linewidth=1, label="distance(anomaly score)")

    ax1.set_ylim(0, 4.2)
    ax1.set_xlim(0, 4000)
    (p2,) = ax2.plot(test_for_plot, "-k", label="test data")

    ax2.set_ylim(0, 10.0)
    ax2.set_xlim(0, 4000)

    # 设置坐标轴的标签
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Distance (anomaly score)", color="blue")
    ax2.set_ylabel("Test Data", color="black")
    # 获取两个轴的图例句柄和标签
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 合并图例句柄和标签
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    # 在 ax1 上显示合并的图例
    ax1.legend(lines, labels)

    # plt.show()
    plt.savefig("knn_result/2.png")  # 保存图像
    plt.close()  # 关闭当前图像


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def embed(lst, dim):
    emb = np.empty((0, dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i : i + dim])[::-1].reshape((1, -1))
        emb = np.append(emb, tmp, axis=0)
    return emb


def plot_data(data):
    # 创建图形
    plt.figure(figsize=(10, 6))
    # 绘制前2000个数据点，颜色为红色
    plt.plot(range(2000), data[0:2000, 2], color="red", label="train data")

    # 绘制后4000个数据点，颜色为黑色
    plt.plot(range(2000, 5999), data[2001:6000, 2], color="black", label="test data")

    # 添加标题和标签
    plt.title("Origional Data")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig("knn_result/1.png")  # 保存图像
    plt.close()  # 关闭当前图像


if __name__ == "__main__":
    main()
