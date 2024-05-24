"""
随机森林，使用了决策树
"""

import numpy as np
import matplotlib.pyplot as plt


# 决策树类
class DecisionTree:
    def __init__(self, split_minimum=2, depth_maximum=100):
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum

    # 树的排列
    def tree_arrangement(self, inputs, node):
        if node.val is not None:
            return node.val
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        return self.tree_arrangement(inputs, node.right)

    # 计算熵
    def entropy(self, target):
        _, hist = np.unique(target, return_counts=True)
        p = hist / len(target)
        return -np.sum(p * np.log2(p))

    # 树的生长
    def tree_growth(self, inputs, target, depth=0):
        samples = inputs.shape[0]
        if depth >= self.depth_maximum or samples < self.split_minimum:
            return Tree_Node(val=np.mean(target))

        thresholds = np.unique(inputs)
        best_gain = -1
        for th in thresholds:
            idx_left = np.where(inputs <= th)
            idx_right = np.where(inputs > th)
            if len(idx_left) == 0 or len(idx_right) == 0:
                gain = 0
            else:
                # ジニ係数という分割基準を計算する
                # 分類問題
                # p1_node1, p2_node1 = probability(target[idx_left])
                # p1_node2, p2_node2 = probability(target[idx_right])
                # sample_sum_node1, sample_sum_node2 = len(idx_left), len(idx_right)
                # gini_node1 = 1 - p1_node1**2 - p2_node1**2
                # gini_node2 = 1 - p1_node2**2 - p2_node2**2
                # gain = weighted_average_gini = gini_node1*(sample_sum_node1 / samples) \
                #     + gini_node2 * (sample_sum_node2 / samples)

                # 回帰問題
                original_entropy = self.entropy(target)
                e_left = self.entropy(target[idx_left])
                e_right = self.entropy(target[idx_right])
                n_left, n_right = len(idx_left), len(idx_right)
                weighted_average_entropy = e_left * (n_left / samples) + e_right * (
                    n_right / samples
                )
                gain = original_entropy - weighted_average_entropy
            if gain > best_gain:
                index_left = idx_left
                index_right = idx_right
                best_gain = gain
                threshhold_best = th

        if best_gain == 0:
            return Tree_Node(val=np.mean(target))

        left_node = self.tree_growth(inputs[index_left], target[index_left], depth + 1)
        right_node = self.tree_growth(
            inputs[index_right], target[index_right], depth + 1
        )
        return Tree_Node(threshhold_best, left_node, right_node)

    # 拟合
    def fit(self, inputs, target):
        self.root_node = self.tree_growth(inputs, target)

    # 预测
    def predict(self, inputs):
        return np.array(
            [self.tree_arrangement(input_, self.root_node) for input_ in inputs]
        )


# 决策树的节点类
class Tree_Node:
    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


# 随机森林类
class RandomForest:
    # 初始化
    def __init__(self, t_numbers=10, split_minimum=5, depth_maximum=100):
        self.t_numbers = t_numbers
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum

    # 拟合
    def fit(self, inputs, target, node_num=10):
        self.use_trees = []
        for _ in range(self.t_numbers):
            tree = DecisionTree(self.split_minimum, self.depth_maximum)
            x_samp, y_samp = self.sampling_bootstrap(inputs, target, node_num)
            tree.fit(x_samp, y_samp)
            self.use_trees.append(tree)

    # 预测
    def predict(self, inputs):
        predicts = np.array([tree.predict(inputs) for tree in self.use_trees])
        return np.mean(predicts, axis=0)

    # 引导法采样
    def sampling_bootstrap(self, inputs, target, node_num):
        idx = np.random.choice(inputs.shape[0], node_num, replace=True)
        return inputs[idx], target[idx]


def main():
    # 准备数据
    # data and plot result
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0, 28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array(
        [62.0, 60.0, 83.0, 120.0, 158.0, 172.0, 167.0, 204.0, 189.0, 140.0, 166.0]
    )

    # 使用RF来预测
    plf = RandomForest(t_numbers=3, depth_maximum=2)  # 3棵树，每棵树2个节点
    plf.fit(inputs, target)
    y_pred = plf.predict(inputs)
    print(y_pred)  # 打印预测结果

    # 数据可视化
    plt.scatter(inputs, target, label="data")
    plt.step(inputs, y_pred, color="orange", label="prediction")
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
