import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels
import io
import requests
import sys

from statsmodels.tsa.arima_model import ARMA

# 月ごとの飛行機の乗客数データ
# 读取数据
url = "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv"
stream = requests.get(url).content
content = pd.read_csv(
    io.StringIO(stream.decode("utf-8")),
    index_col="Month",  # 将 month 设置为索引
    parse_dates=True,  # 解析为日期
    dtype="float",
)

passengers = content["#Passengers"][:120]  # 提取前120行数据
passengers_plot = content["#Passengers"]
# passengers= np.diff(np.log(passengers))
# 绘制完整的乘客数时间序列图
plt.plot(passengers_plot)
plt.title("passengers data")


# 使用ADF检验 passengers 的数据
"""
ADF检验的原假设是时间序列存在单位根，即非定常
通过计算ADF统计量，并且与临界值进行比较，可以判断是否拒绝原假设
"""
result = sm.tsa.stattools.adfuller(passengers)
# 打印 ADF 的统计量
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
print("Critical Values:")
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))
# 绘制自相关函数ACF 和 偏自相关函数PACF图，定性分析时间序列的相关性结构
sm.graphics.tsa.plot_acf(passengers, lags=50)
sm.graphics.tsa.plot_pacf(passengers, lags=50)


ar = sm.tsa.AR(passengers)
print(
    "the order of arma is", ar.select_order(maxlag=6, ic="aic")
)  # 打印 ARMA 模型的最佳阶数
AR = ARMA(passengers, order=(5, 0)).fit(dist=False)

# 提取模型残差，并绘制自相关函数和 偏自相关函数，检验模型残差的白噪声性质
resid = AR.resid
fig = plt.figure(figsize=(5, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


# 预测
pred = AR.predict("1955-01-01", "1958-12-01")
plt.figure(figsize=(6, 5))
plt.plot(passengers[70:120], "--")
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = AR.predict("1958-01-01", "1965-12-01")
plt.figure(figsize=(4, 5))
plt.plot(passengers[40:], "--")
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()

"""
pred = AR.predict('1950-01-01', '1953-12-01')
plt.figure(figsize=(6,5))
plt.plot(passengers[10:60],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)


pred = AR.predict('1958-01-01', '1965-12-01')
plt.figure(figsize=(4,5))
plt.plot(passengers[40:],'--')
plt.plot(pred, "k")
plt.xticks(rotation=45)

plt.show()
"""
