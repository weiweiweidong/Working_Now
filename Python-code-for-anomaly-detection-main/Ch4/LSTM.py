########################## 这部分是为了设置相对路径而作的改动 ##########################
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前脚本所在目录
os.chdir(script_dir)
# 打印当前工作目录以确认
print("Current working directory:", os.getcwd())
##################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

import renom as rm
from renom.optimizer import Adam
from renom.cuda import set_cuda_active

set_cuda_active(True)


# データの前処理
# 数据记载和预处理
# df = pd.read_csv('data/qtdbsel102.txt', header=None, delimiter='\t')
df = pd.read_csv("qtdbsel102.txt", header=None, delimiter="\t")
ecg = df.iloc[:, 2].values  # 加载数据并提取第3列作为 ECG 数据
ecg = ecg.reshape(len(ecg), -1)
print("length of ECG data : ", len(ecg))
print(ecg.shape)

# standardize 标准化处理：将数据转化为均值为1，方差为1的分布
scaler = StandardScaler()
std_ecg = scaler.fit_transform(ecg)  # 将ecg数据标准化


# 可视化标准化后的 ECG 信号
# plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.xlabel("time")
plt.ylabel("ECG's value")
plt.plot(np.arange(5000), std_ecg[:5000], color="black")  # 显示前5000个数据
plt.ylim(-3, 3)
plt.tick_params(top=1, right=1, direction="in")
# x = np.arange(4200,4400)
# y1 = [-3]*len(x)
# y2 = [3]*len(x)
# plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
# plt.show()
plt.savefig("LSTM_result/1.png")  # 保存图像
plt.close()  # 关闭当前图像

normal_cycle = std_ecg[5000:]  # 提取5000以后的数据

plt.figure(figsize=(10, 5))
plt.tick_params(top=1, right=1, direction="in")
# plt.title("training data")
plt.xlabel("time")
plt.ylabel("ECG's value")
plt.plot(
    np.arange(5000, 8000), normal_cycle[:3000], color="black"
)  # stop plot at 8000 times for friendly visual    打印5000～8000之间的数据
# plt.show()
plt.savefig("LSTM_result/2.png")  # 保存图像
plt.close()  # 关闭当前图像


# 時系列「ts」から「look_back」の長さのデータを作成します
# 划分子序列
def create_subseq(ts, look_back, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts) - look_back - pred_length):
        sub_seq.append(ts[i : i + look_back])
        next_values.append(ts[i + look_back : i + look_back + pred_length].T[0])
    return sub_seq, next_values


look_back = 10  # 子序列长度
pred_length = 3  # 预测的长度
sub_seq, next_values = create_subseq(normal_cycle, look_back, pred_length)
X_train, X_test, y_train, y_test = train_test_split(
    sub_seq, next_values, test_size=0.2
)  # 划分训练集和测试集，比例为 80:20
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


train_size = X_train.shape[0]
test_size = X_test.shape[0]
print("train size:{}, test size:{}".format(train_size, test_size))


# モデルの定義、学習
# 定义模型
"""
由两层LSTM和两层Relu激活函数组成的序列模型，并使用Adam优化器进行训练。训练过程中使用批量梯度下降法，每经过一定次数的epoch后进行早停检查。如果测试损失没有显著改善则停止训练。
双层LSTM网络可以捕捉更复杂的模式和特征。第一层LSTM捕捉低级别的时间依赖关系，而第二层LSTM在第一层的基础上捕捉更高级别的时间依赖关系。这样可以提高模型对复杂模式的理解和预测能力。
"""
model = rm.Sequential(
    [rm.Lstm(35), rm.Relu(), rm.Lstm(35), rm.Relu(), rm.Dense(pred_length)]
)

# パラメータ
batch_size = 100  # 每个批次batch包含100组数据
max_epoch = 2000
period = 10  # early stopping checking period   每隔10epoch进行一次早停检查

optimizer = Adam()  # 是用 Adam 优化器
epoch = 0
loss_prev = np.inf  # 前一个epoch的验证损失初始化为无穷大（设置初始值）
learning_curve, test_curve = [], []
while epoch < max_epoch:
    epoch += 1
    # 生成一个从0～train_size-1 的随机排列数组，打乱训练数据
    # 例如：生成一个10的随机排列数组：[3 1 6 0 5 2 7 9 4 8]
    perm = np.random.permutation(train_size)
    # 设置loss为0
    train_loss = 0
    # 按batch大小进行循环，直到遍历完所有数据
    for i in range(train_size // batch_size):
        batch_x = X_train[perm[i * batch_size : (i + 1) * batch_size]]
        batch_y = y_train[perm[i * batch_size : (i + 1) * batch_size]]
        l = 0
        z = 0
        # 切换到训练模式
        with model.train():
            for t in range(look_back):
                z = model(batch_x[:, t])  # 计算当前时间步的模型输出
                l = rm.mse(z, batch_y)  # 计算当前时间步的MSE loss
            model.truncate()  # 清除模型的隐藏状态，准备下一批数据。
        l.grad().update(optimizer)  # 更新模型参数
        train_loss += l.as_ndarray()  # 将当前的loss累加到总loss里
    train_loss /= train_size // batch_size  # 计算当前 epoch 的平均训练损失
    learning_curve.append(train_loss)  # 将当前 epoch 的 train loss 添加到学习曲线中

    # test  测试模型
    l = 0
    z = 0
    for t in range(look_back):
        z = model(X_test[:, t])  # 使用测试集，计算当前模型的预测值
        l = rm.mse(z, y_test)  # 计算 MSE loss
    model.truncate()  # 清除模型的隐藏状态，准备下一批数据。
    test_loss = l.as_ndarray()  # 将当前的 test loss 累加到总 loss 里
    test_curve.append(test_loss)

    # check early stopping
    if epoch % period == 0:  # 如果当前epoch 是period 的倍数
        # 打印 loss
        print(
            "epoch:{} train loss:{} test loss:{}".format(epoch, train_loss, test_loss)
        )
        # 计算当前的 test loss 是否比上一轮的loss只少了一点点（1%）
        if test_loss > loss_prev * 0.99:
            # 如果在 1% 以内，就说明loss不再变化了，早停
            print("Stop learning")
            break
        else:
            loss_prev = deepcopy(test_loss)

# 打印学习曲线
plt.figure(figsize=(10, 5))
plt.plot(learning_curve, color="black", label="learning curve")
plt.plot(test_curve, color="red", label="test curve")
plt.tick_params(top=1, right=1, direction="in")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim([0.0, 0.01])
plt.legend()
# plt.show()
plt.savefig("LSTM_result/3.png")  # 保存图像
plt.close()  # 关闭当前图像


# 定常性確認
# 定常性验证
sub_seq, next_values = create_subseq(
    std_ecg[5000:8000], look_back, pred_length
)  # 截取5000～8000之间的数据
sub_seq = np.array(sub_seq)
next_values = np.array(next_values)

# 对截取出来的数据进行预测
for t in range(look_back):
    pred = model(sub_seq[:, t])
model.truncate()
errors = next_values - pred  # 计算误差

print(next_values.shape)  # 打印label的size：(2987,3)
print(pred.shape)


row_y = []
pred_y = []

row_y.append(next_values[0, 0])
pred_y.append(pred[0, 0])
row_y.append((next_values[0, 1] + next_values[1, 0]) / 2)
pred_y.append((pred[0, 1] + pred[1, 0]) / 2)
for i in range(2987 - 4):
    row_y.append(
        (next_values[i, 2] + next_values[i + 1, 1] + next_values[i + 2, 0]) / 3
    )
    pred_y.append((pred[i, 2] + pred[i + 1, 1] + pred[i + 2, 0]) / 3)
row_y.append((next_values[-2, 2] + next_values[-2, 1]) / 2)
pred_y.append((pred[-2, 2] + pred[-2, 1]) / 2)
row_y.append(next_values[-1, 2])
pred_y.append(pred[-1, 2])

row_y = np.array(row_y)
pred_y = np.array(pred_y)

print(row_y.shape)
print(pred_y.shape)

plt.figure(figsize=(10, 5))
plt.tick_params(top=1, right=1, direction="in")
plt.xlabel("time")
plt.ylabel("ECG's value")
plt.plot(np.arange(5013, 8000), row_y, color="black", label="row data")
plt.plot(np.arange(5013, 8000), pred_y, color="cyan", label="pred data")
plt.xlim([6000, 7000])
plt.legend()
# plt.show()
plt.savefig("LSTM_result/4.png")  # 保存图像
plt.close()  # 关闭当前图像


# フィッティング
# 计算测试集合的 预测结果误差的均值和协方差
for t in range(look_back):
    pred = model(X_test[:, t])
model.truncate()
errors = y_test - pred
mean = sum(errors) / len(errors)
cov = 0
for e in errors:
    cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
cov /= len(errors)

print("mean : ", mean)
print("cov : ", cov)


# 计算马氏距离：Mahalanobis距离用来表示 点 与 分布中心之间的距离
# マハラノビス距離を計算する
def Mahala_distantce(x, mean, cov):
    d = np.dot(x - mean, np.linalg.inv(cov))
    d = np.dot(d, (x - mean).T)
    return d


# 異常検知を実行する
# 进行异常检测
sub_seq, next_values = create_subseq(
    std_ecg[:5000], look_back, pred_length
)  # 拿出来前5000个数据
sub_seq = np.array(sub_seq)
next_values = np.array(next_values)
for t in range(look_back):
    pred = model(sub_seq[:, t])
model.truncate()
errors = next_values - pred  # 计算误差

m_dist = [0] * look_back
for e in errors:
    m_dist.append(Mahala_distantce(e, mean, cov))  # 计算马氏距离

plt.hist(m_dist, bins=1000, color="black")
plt.tick_params(top=1, right=1, direction="in")
plt.ylabel("Number of data")
plt.xlabel("Mahalanobis Distance")
plt.xlim([-50, 1600])
plt.ylim([0, 700])
# plt.show()
plt.savefig("LSTM_result/5.png")  # 保存图像
plt.close()  # 关闭当前图像
# print(m_dist)


m_dist_sort = sorted(m_dist, reverse=True)  # 对马氏距离降序排序
print(m_dist_sort[149])  # 打印第150个马氏距离的值：45.572

m_dist = [0] * look_back
for e in errors:
    m_dist.append(Mahala_distantce(e, mean, cov))

fig, axes = plt.subplots(nrows=2, figsize=(15, 10))  # 创建一个包含两个子图的图表

# 第一个子图：绘制原始的 ecg 图像
axes[0].plot(std_ecg[:5000], color="black", label="original data")
axes[0].set_xlabel("time")
axes[0].set_ylabel("ECG's value")
axes[0].set_ylim(-3, 3)
x = np.arange(4200, 4400)
axes[0].tick_params(top=1, right=1, direction="in")
# y1 = [-3]*len(x)
# y2 = [3]*len(x)
# axes[0].fill_between(x, y1, y2, facecolor='g', alpha=.3)

# 第二个子图：绘制马氏距离
axes[1].plot(m_dist, color="r", label="Mahalanobis Distance")
axes[1].set_xlabel("time")
axes[1].set_ylabel("Mahalanobis Distance")
axes[1].set_ylim(0, 1000)
th = 32.15269  # 设置阈值
axes[1].plot(
    [0, 5000], [th, th], color="black", linestyle="-", label="threshold", linewidth=1
)

# axes[1].tick_params(top=1, right=1, direction='in')
# y1 = [0]*len(x)
# y2 = [1000]*len(x)
# axes[1].fill_between(x, y1, y2, facecolor='g', alpha=.3)

plt.legend()
# plt.show()
plt.savefig("LSTM_result/6.png")  # 保存图像
plt.close()  # 关闭当前图像
