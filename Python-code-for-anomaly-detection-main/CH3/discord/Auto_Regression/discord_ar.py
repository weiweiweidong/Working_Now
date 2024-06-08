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
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
import random
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa import stattools as st
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


# np.random.seed(1273465)


# 加载数据：1～6000行
def load_data():
    df_read = pd.read_csv("discord.csv")
    return df_read.iloc[1:6000]


df_nile = load_data()


df_read = sm.datasets.nile.load_pandas().data
# print (df_read['volume'])


# Data Split (70: 30)
# 拆分数据：1～4000 是train集，4000～6000 是test集
df_test = df_nile["volume"].iloc[4000:6001]
df_train = df_nile["volume"].iloc[:4000]


# from statsmodels.tsa import stattools as st
# ARMAモデルの次数を決める
# 使用 BIC（贝叶斯信息准则）确定ARMA模型的最佳阶数
print(st.arma_order_select_ic(df_train, ic="bic", trend="nc"))

arma_11 = sm.tsa.ARMA(df_train, order=(3, 0)).fit()  # 构建和训练一个ARMA模型

# arma_11 = sm.tsa.SARIMAX(
#     df_train,
#     order=(3, 1, 2),
#     seasonal_order=(0, 0, 0, 213),
#     enforce_stationarity=False,
#     enforce_invertibility=False,
# ).fit()

# in-sample predict 进行样本内预测？？？？
arma_11_inpred = arma_11.predict(start=2, end=4000, typ="levels")
# out-of-sample predict 进行样本外预测？？？？
arma_11_outpred = arma_11.predict(start=3999, end=6000, typ="levels")


# plot data and predicted values
# def plot_ARMA_results(origdata, pred11in, pred11out):
#     # 绘制原始数据
#     ax = origdata["volume"].plot(
#         figsize=(14, 10),
#         grid=False,
#         color="g",
#         # marker="o",
#         # markersize=2,
#         # markerfacecolor="w",
#     )
#     # pred11in.plot(color=["b"], linestyle="dotted")    # 按照点方式绘图
#     pred11in.plot(color=["b"])
#     pred11in1 = pred11in - df_train + 3.0  # 样本内的预测
#     pred11out1 = pred11out - df_test + 3.0  # 样本外的预测
#     pred11out.plot(color=["r"])
#     pred11out1.plot(color=["r"], linestyle="dotted")
#     pred11in1.plot(color=["b"], linestyle="dotted", alpha=0.5)
#     ax.set_xlabel("mili second")
#     ax.set_ylabel("ms^2/Hz")
#     ax.set_ylim(1, 7)


def plot_ARMA_results(origdata, pred11in, pred11out):
    # 绘制预测数据
    pred11in.plot(color=["b"], linewidth=1.5, label="In-sample Prediction")
    pred11in1 = pred11in - df_train + 3.0  # 样本内的预测
    pred11out1 = pred11out - df_test + 3.0  # 样本外的预测
    pred11out.plot(color=["r"], linewidth=1.5, label="Out-of-sample Prediction")
    pred11out1.plot(
        color=["r"], linestyle="dotted", linewidth=1, label="Out-of-sample Error"
    )
    pred11in1.plot(
        color=["b"], linestyle="dotted", alpha=0.5, linewidth=1, label="in-sample Error"
    )

    # 最后绘制原始数据，以确保显示在最前面
    ax = origdata["volume"].plot(
        figsize=(14, 8),
        grid=False,
        color="k",
        linewidth=1,  # 设置原始数据的线宽
        label="origional Data",
        # marker="o",
        # markersize=2,
        # markerfacecolor="w",
    )

    ax.set_xlabel("mili second")
    ax.set_ylabel("ms^2/Hz")
    ax.set_ylim(1, 7)


# call the plot
plot_ARMA_results(df_nile, arma_11_inpred, arma_11_outpred)

# 计算预测误差的排序，找到第20大的误差值作为阈值
abnormalty_sort = np.sort(arma_11_outpred - df_test)
np.set_printoptions(threshold=np.inf)
th = abnormalty_sort[::-1][20]
print("a is ", th)


th = th + 3
plt.plot(
    [4000, 6000], [th, th], color="m", linestyle="-", linewidth=0.5, label="threshold"
)

plt.legend()
# plt.show()
plt.savefig("ar_result/1.png")  # 保存图像
plt.close()  # 关闭当前图像
