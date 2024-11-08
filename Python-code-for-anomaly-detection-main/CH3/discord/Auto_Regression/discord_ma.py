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


def load_data():
    df_read = pd.read_csv("discord.csv")
    return df_read.iloc[1:6000]


df_nile = load_data()


df_read = sm.datasets.nile.load_pandas().data
# print (df_read['volume'])

# Data Split (70: 30)


df_test = df_nile["volume"].iloc[4000:6001]  # 4000～6000 作为test集
df_train = df_nile["volume"].iloc[:4000]  # 0～4000 作为train集
df_train_all = df_nile["volume"].iloc[:6000]

arma_11 = sm.tsa.ARMA(df_train, (0, 1)).fit()  # AR部分阶数为0，MA部分阶数为1

arma_11_all = sm.tsa.ARMA(df_train_all, (0, 1)).fit()  # 训练

arma_11_inpred = arma_11.predict(start=2, end=4000, typ="levels")  # in-sample 预测
# out-of-sample predict
arma_11_outpred = arma_11.predict(
    start=3999, end=6000, typ="levels"
)  # out-of-sample 预测
# plot data and predicted values
arma_11_pred_all = arma_11_all.predict(typ="levels")
# plot data and predicted values


def plot_ARMA_results(origdata, pred11in, pred11out):
    ax = origdata["volume"].plot(
        figsize=(14, 10),
        grid=False,
        color="k",
        # marker="o",
        # markersize=2,
        # markerfacecolor="w",
    )
    pred11in.plot(color=["b"], linestyle="dotted")
    pred11in1 = pred11in - df_train + 3.0
    pred11out1 = pred11out - df_test + 3.0
    pred11out.plot(color=["r"])
    pred11out1.plot(color=["r"], linestyle="dotted")
    pred11in1.plot(color=["b"], linestyle="dotted")
    ax.set_xlabel("mili second")
    ax.set_ylabel("ms^2/Hz")
    ax.set_ylim(1, 7)
    plt.show()


"""
def plot_ARMA_results(origdata, pred11):
    ax = origdata['volume'].plot(figsize=(10,2), grid=False,color='k',marker ='o', markersize=2,markerfacecolor='w')
    pred11.plot(color=['b'],linestyle='dotted')
    pred11in1= pred11-df_train_all+3.0

    pred11in1.plot(color=['b'],linestyle='dotted')

    ax.set_xlabel('mili second')
    ax.set_ylabel('ms^2/Hz')
    ax.set_ylim(1,7)
    plt.show()

#call the plot

plot_ARMA_results(df_nile, arma_11_pred_all)
"""

plot_ARMA_results(df_nile, arma_11_inpred, arma_11_outpred)

print(df_nile["volume"].values)
print(arma_11_inpred.values)


plt.show()


# df_nile['volume'].to_csv('1.csv')
# arma_11_inpred.to_csv('2.csv')
