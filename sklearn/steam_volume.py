# 工业蒸汽量预测

import pandas as pd
import numpy as np
# 它用于创建 2D 和 3D 图形
import matplotlib.pyplot as plt
# 它用于创建统计图形，Seaborn 建立在 Matplotlib 之上，并提供了一个高级界面来创建有吸引力和信息丰富的统计图形
import seaborn as sns
# learning_curve学习曲线可以清晰的看出模型对数据的过拟合和欠拟合
from sklearn.model_selection import train_test_split, learning_curve
# 均方误差（MSE）：真实值和预测值差距，越小越好
from sklearn.metrics import mean_squared_error
# 线性回归模型
from sklearn.linear_model import LinearRegression
# 线性支持向量回归，函数y = wx + b，和线性回归区别：计算损失的原则不同，目标函数和最优化算法也不同
from sklearn.svm import LinearSVR
# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

# 数据读取
train_data = pd.read_table("zhengqi_train.txt")
test_data = pd.read_table("zhengqi_test.txt")
# 查看是否有空白区域，有就去掉
print(train_data.isnull().sum())

# 数据划分：训练数据和标签
# 剔除target这1列所有数据
train_data_X = train_data.drop(['target'], axis=1)
train_data_y = train_data['target']

# 比较特征变量
# 对train和test中的38个变量特征分布进行比较，通过分布图对特征进行去除无关变量
plt.figure(figsize=(30, 30))
i = 1
for col in test_data.columns:
    # 描述子图的位置：行、列、索引；5行8列，索引从1开始
    plt.subplot(5, 8, i)
    # 默认绘制直方图并拟合内核密度估计
    sns.distplot(train_data_X[col], color='red')
    sns.distplot(test_data[col], color='blue')
    # 设置图例
    plt.legend(['Train', 'Test'])
    i += 1

plt.show()
plt.clf()

# 删除差异较大变量
# 由上图可知以下14个变量训练集和测试集的特征差异较大，则删除
train_data_X_new = train_data_X.drop(['V2', 'V5', 'V9', 'V11', 'V13', 'V14', 'V17', 'V19', 'V20', 'V21', 'V22', 'V24', 'V27', 'V35'], axis=1)
test_data_new = test_data.drop(['V2', 'V5', 'V9', 'V11', 'V13', 'V14', 'V17', 'V19', 'V20', 'V21', 'V22', 'V24', 'V27', 'V35'], axis=1)
# all_data_X = pd.concat([train_data_X_new, test_data_new])

# 数据集的切割：分割训练集和测试集
# X_train训练数据，y_train训练数据标签，X_test测试数据, y_test测试数据标签
# random_state：随机状态，如果不设置，拆分的结果随机
X_train, X_test, y_train, y_test = train_test_split(train_data_X_new, train_data_y, test_size=0.3, random_state=827)


# 线性回归：
def Linear_Regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 预测测试集数据
    res = model.predict(X_test)
    # 计算预测值和真实值的差距，越小越好
    mse = mean_squared_error(y_test, res)

    print('Linear_Regression训练集得分：{}'.format(model.score(X_train, y_train)))
    print('Linear_Regression测试集得分：{}'.format(model.score(X_test, y_test)))
    print('Linear_Regression测试集的MSE得分：{}'.format(mse))
    print('--------------------------------')


# 向量机
def Linear_SVR(X_train, X_test, y_train, y_test):
    model = LinearSVR()  # 可以调整参数C乘法系数，提高准确率
    model.fit(X_train, y_train)
    res = model.predict(X_test)
    mse = mean_squared_error(y_test, res)

    print('Linear_SVR训练集得分：{}'.format(model.score(X_train, y_train)))
    print('Linera_SVR测试集得分：{}'.format(model.score(X_test, y_test)))
    print('Linear_SVR测试集MSE得分：{}'.format(mse))
    print('------------------------------')


Linear_Regression(X_train, X_test, y_train, y_test)
Linear_SVR(X_train, X_test, y_train, y_test)


# 算法融合
# 定义线性回归模型，并得到预测值
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_res = lr_model.predict(X_test)

# 定义SVM模型，得到预测值
svm = LinearSVR()
svm.fit(X_train, y_train)
svm_res = svm.predict(X_test)

# 将两个算法的预测值，存入pandas类里面
new_data = pd.DataFrame({'lr': lr_res, 'svm': svm_res})

# 这里为什么划分 测试集和测试标签，而不是训练集?
new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_data, y_test, test_size=0.3, random_state=827)
mix_lr = LinearRegression()
mix_lr.fit(new_x_train, new_y_train)
new_res = mix_lr.predict(new_x_test)
mse = mean_squared_error(new_y_test, new_res)
print('model训练集得分：{}'.format(mix_lr.score(new_x_train, new_y_train)))
print('model测试集MSE得分：', mse)  # 融合以后差值更大?
print('------------------------------')


# 因为融合的差值比较大，所以进行以下权重调整?
# 2个模型融合
def model_mix(pred_1, pred_2):
    result = pd.DataFrame(columns=['LinearRegression', 'SVR', 'Combine'])

    # i = 0; i < 20; i += 1
    # 这个地方为何设置20，10是否也行?
    for a in range(1, 20, 1):
        for b in range(1, 20, 1):
            # 通过增加权重融合2种算法，得到融合后的预测值
            # 某种数学公式?
            y_pred = (a*pred_1+b*pred_2) / (a+b)
            # 计算融合算法的预测值和真实值之间的差距
            mse = mean_squared_error(y_test, y_pred)
            # 将两个权重a,b 和 差值存入列表中
            # append添加字典为何数据类型变成了float?
            # ignore_index = True：不使用索引标签，如果为false将会往后追加列，为true会添加到相同的列上，这个地方可以省略
            result = result.append([{'LinearRegression': a,
                                     'SVR': b,
                                     'Combine': mse}],
                                   ignore_index=True)
    return result


model_combine = model_mix(lr_res, svm_res)
# 根据融合算法的mse（预测值和真实值的差距）排序，获取结果列表中最小的mse的组合
# ascending：True升序（默认）、False降序
# inplace：是否用排序后的数据集替换原来的数据集，默认False
model_combine.sort_values(by='Combine', inplace=True)
# 显示数据前几行，参数为数值，默认5行
print(model_combine.head())

# 得到最优的权重a、b，如何使用?


