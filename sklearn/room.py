from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据集
data = datasets.load_boston()

# CRIM     城镇人均犯罪率
# ZN       占地面积超过2.5万平方英尺的住宅用地比例
# INDUS    城镇非零售业务地区的比例
# CHAS     查尔斯河虚拟变量 (= 1 如果土地在河边；否则是0)
# NOX      一氧化氮浓度（每千万）
# RM       平均每居民房数
# AGE      在1940年之前建成的所有者占用单位的比例
# DIS      与五个波士顿就业中心的加权距离
# RAD      辐射状公路的可达性指数
# TAX      每10000美元的全额物业税率
# PTRATIO  城镇师生比例
# B        1000(Bk-0.63)²，其中 Bk 是城镇黑人的比例
# LSTAT    低地位人口的百分比
# MEDV     以1000美元计算的自有住房的中位数

d = pd.DataFrame(data.data, columns=data.feature_names)
print(d)

# 5表示只取第6列（RM-平均每居民房数）
x = data.data[:, 5]
y = data.target
# 填充画布
# figsize = 长,宽
plt.figure(figsize=(10, 7))
# 绘制散点图
# x、y：输入数据，相同长度的数组
# c：颜色序列，可以是颜色的首字母
# marker：样式
plt.scatter(x, y, c='r', marker='*')
# 绘制直线
# x为：x轴坐标，y为：y轴坐标，x可以省略（默认0、1、2...）
# o为圆点、g为绿色、:号为点线
# 为何是这种线性方程式?
plt.plot(x, 7*x+10, 'og:')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data.data[:, 5], data.target, test_size=0.3)
# 线性回归算法：
# 机器学习所针对的问题有两种：一种是回归，一种是分类（KNN）。回归是解决连续数据的预测问题，而分类是解决离散数据的预测问题。
lr = LinearRegression()
# fit第1个参数必须是二维数组，第二个参数一般是一维数组，行数必须和第一个参数相同
# 因为前面train_test_split中data.data[:, 5]只取第6列，所以x_train为一维数组，因此需要转换
# reshape 354列（一维数组）转变为 354行（二维数组，354行*1列）
lr.fit(x_train.reshape(-1, 1), y_train)
# 直线方程y=kx+b中，b就是截距，k就是斜率，房间数量x，y就表示房价
print('斜率', lr.coef_, '截距', lr.intercept_)
print('-------------------------------------------')

# 使用斜率和截距做为直线方程式
plt.scatter(x, y, c='r', marker='*')
plt.plot(x, lr.coef_*x+lr.intercept_, 'g')
plt.show()


# 看整体数据预测结果
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print('整体斜率：', mlr.coef_)
print('整体截距：', mlr.intercept_)
print('-------------------------------------------')
x_predict = mlr.predict(x_test)
print("预测结果:", x_predict[: 10])
print("真实结果:", y_test[: 10])
print('-------------------------------------------')

# 这个方程式有点像 predict 内部计算方式
a = np.sum(x_test[0]*mlr.coef_)+mlr.intercept_
# 计算结果 和 预测结果一致
print(a, x_predict[0])
print('-------------------------------------------')

# 对拟合效果打分（0-1之间）
# R2=(1-u/v)、 u=((y_true - y_pred) ** 2).sum()、 v=((y_true-y_true.mean())**2).sum()
# y_true和y_pred越接近，u/v的值越小，R2的值就越大, 成绩越好
print('最后评分：', mlr.score(x_test, y_test))

# 准确率
# 预测值和实际值 正负相差5%内 为准确
a = ((x_predict - y_test) / y_test)
a_correct = a[np.abs(a) <= 0.05]
s = len(a_correct) / len(a) * 100
print('准确率：', s)

# 如何提升模型的准确率?




