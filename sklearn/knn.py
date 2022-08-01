from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 引入KNN模型
from sklearn.neighbors import KNeighborsClassifier
# 交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 近邻分类（K-近邻算法）
# 思想：首先要将这个样本与训练样本集中的每个样本计算距离或相似度（使用欧氏距离），找出与该样本最近的或最相似的K个训练样本，对这K个训练样本的类别进行统计，选择其中类别数最多的作为这个待分类样本的类别。

# 引入训练数据 和 测试数据
iris = load_iris()
x = iris.data
y = iris.target
# 萼片长度（cm）, 萼片宽度（cm）, 花瓣长度（cm）, 花瓣宽度（cm）
print(iris.feature_names)

# 将数据分割 训练集 和 测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape)
print(y_train.shape)

# 初始化KNN模型：
# n_neighbors：表示k-nn算法中选取离测试数据最近的k个点
knn = KNeighborsClassifier(n_neighbors=5)


# 机器学习对数据的处理就是分隔训练集和测试集，用训练集去训练模型，用测试集去测试模型的性能或是否过拟合，
# 但这样做有点不可靠，万一不好的模型对测试集过拟合，这样我们就错误认为模型是好的，这就需要使用交叉验证，
# ** 交叉验证就是用来判断模型拟合的好坏 **

# K折交叉验证：将训练数据D分为K份，用其中（K-1）份训练模型，剩余1份数据用于评估模型的质量
# 例如以下cv= 5折交叉验证，将训练数据x分为：x1、x2...x5，每次取其中4份数据做为训练集，1份做为测试集，
# 最终将循环后所有的评估结果取平均，详情：https://zhuanlan.zhihu.com/p/31924220

# x：样本数据
# y：样本标签
# cv：交叉验证生成器 或 交叉验证折数
# soring：调用的方法，这里是准确率，详情：https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
# 因为cv=5，所以评估结果为5个数
print("评估结果：", scores)
# 把k次评估指标的平均值做为最终的评估指标（拟合的好坏），越大越好
print("评估均值：", scores.mean())


# 训练
knn.fit(x_train, y_train)
# 预测
y_predoet = knn.predict(x_test)
print("预测结果:", y_predoet)
print("真实结果:", y_test)


# 过于拟合时，如何选择最优的k：
# 循环n_neighbors 1-30，查找出均值最高的k
# 从图形中看出：k=29时，均值最小; k=10时，均值最大（最优）
# 因此我们将n_neighbors设置成10时准确率最高
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    # knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
    kn = KNeighborsClassifier(k)
    score = cross_val_score(kn, x, y, cv=5, scoring='accuracy')
    cv_scores.append(score.mean())

plt.plot(k_range, cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()