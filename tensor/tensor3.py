import tensorflow as tf
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.max(x_train))

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(np.max(x_train))

# 建立模型，训练数据（每张图片的大小是28*28；神经网络有128层，最后要从0-9共10个数）
model = tf.keras.models.Sequential()
# 输入层
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 隐藏层-1 784*128
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
# 隐藏层-2 128*128
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# 隐藏层-2 128*128
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# 输出层 128*10
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.summary()

# 设置优化器、损失函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型，3轮
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("测试集准确率：", val_acc)

# 预测，并查看一下预测数据集的前25个, 并且用图形展示一下跟实际值对比
predictions = model.predict(x_test)
pred = np.argmax(predictions[:25, :], axis=1)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel("true: {} / pred: {}".format(y_test[i], pred[i]))
plt.show()


# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# print(y_train[0])