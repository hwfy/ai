from sklearn import datasets
import pandas as pd
import numpy as np
# 画图
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split


# 数据集
data = datasets.load_boston()

d = pd.DataFrame(data.data, columns=data.feature_names)
print(d)

# x = data.data[:, 5]
# y = data.target
# # 填充画布 长/宽
# plt.figure(figsize=(10, 7))
# # 画点
# plt.scatter(x, y, c='r', marker='*')
# # 画线
# plt.plot(x, 7*x+10, 'g')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(data.data[:, 5], data.target, test_size=0.3)
lr = LinearRegression()
lr.fit(x_train.reshape(-1, 1), y_train)

print('斜率', lr.coef_, '截距', lr.intercept_)

# x= data.data[:, 5]
# y= data.target
# plt.figure(figsize=(10, 7))
# # Z = np.random.rand(10, 12)
# # plt.pcolor(Z)
# # 画点
# plt.scatter(x, y, c='r', marker='*')
# # 画线
# plt.plot(x, lr.coef_*x+lr.intercept_, 'g')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print(mlr.coef_, mlr.intercept_)

x_predict = mlr.predict(x_test)
print("预测结果:", x_predict[0: 10])
print("真实结果:", y_test[: 10])

a = np.sum(x_test[0]*mlr.coef_)+mlr.intercept_
print(a, x_predict[0])

# R2=(1-u/v) u=((y_true - y_pred) ** 2).sum() v=((y_true-y_true.mean())**2).sum()
print(mlr.score(x_test, y_test))

# 准确率
# 预测值和实际值 正负相差5%内 为准确
a = ((x_predict - y_test) / y_test)
a_correct = a[np.abs(a) <= 0.05]
s = len(a_correct)/len(a)*100
print(s)




