"""
对Davis数据集的异常检测和可视化
"""

# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from numpy import linalg as la
from scipy.stats import norm
import seaborn as sns

########################## 这部分是为了设置相对路径而作的改动 ##########################
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前脚本所在目录
os.chdir(script_dir)
# 打印当前工作目录以确认
print("Current working directory:", os.getcwd())
##################################################################################

davis = pd.read_csv("./data/Davis.csv").values
x = davis[:, 2:3]
# ヒストグラム分布を作る
sns.distplot(x, fit=norm, color="k", kde=False, bins=50, rug=True)
plt.show()
# 平均ベクトル
mx = x.mean(axis=0)
# 中心化データ
xc = x - mx


# 平均値からの誤差
xc = x - mx
# 誤差のばらつき
sx = (xc.T.dot(xc) / x[:, 0].size).astype(float)
# 誤差と誤差のばらつきによる異常度の定義
ap = np.dot(xc, np.linalg.inv(sx)) * xc
plt.hist(ap, color="blue", bins=200)
plt.show()
# 閾値:分位点法
th = 4.27
plt.scatter(np.arange(ap.size), ap, color="b")
plt.plot([0, 200], [th, th], color="red", linestyle="-", linewidth=1)
plt.ylim(0, 55)
plt.show()
