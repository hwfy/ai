from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 引入KNN模型
from sklearn.neighbors import KNeighborsClassifier
# 交叉验证
from sklearn.model_selection import cross_val_score

# 近邻分类
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

# 初始化KNN模型
knn = KNeighborsClassifier(n_neighbors=5)

# 准确率
scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
print(scores)
# 均值
print(scores.mean())

# 训练
knn.fit(x_train, y_train)

# 预测
y_predoet = knn.predict(x_test)

print("预测结果:", y_predoet)
print("真实结果:", y_test)