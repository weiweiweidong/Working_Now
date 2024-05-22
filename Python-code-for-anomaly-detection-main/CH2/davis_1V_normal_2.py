# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from numpy import linalg

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
# 平均ベクトル
mx = x.mean(axis=0)
# 中心化データ
xc = x - mx
# 標本分散ベクトル
sx = (xc.T.dot(xc) / x[:, 0].size).astype(float)
# 異常度
ap = np.dot(xc, np.linalg.inv(sx)) * xc
# 閾値
th = sp.stats.chi2.ppf(0.98, 1)
plt.scatter(np.arange(ap.size), ap, color="b")
plt.plot([0, 200], [th, th], color="red", linestyle="-", linewidth=1)
plt.ylim(0, 55)
plt.show()
