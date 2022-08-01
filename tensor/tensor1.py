from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 28x28大小的0-9数字图像的手写体数据集（单通道的灰度图片）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 加快机器学习速度：将0-255的整数标准化为0-1的范围之间的浮点数
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# 建立模型，训练数据（每张图片的大小是28*28；神经网络有128层，最后要从0-9共10个数）
# Sequential模型可以构建非常复杂的神经网络，包括全连接神经网络、卷积神经网络(CNN)、循环神经网络(RNN)等等
# 这里创建的是全连接神经网络，因为没有Conv2D卷积 和 MaxPooling2D池化层
model = tf.keras.models.Sequential()
# 输入层，28*28 = (0, 784)，把二维的图像数据一维化
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 隐藏层-1 784*1024，Dense用于创建1024个神经元的全连接层，一般使用ReLU作为激活函数
# 全连接层中的每个神经元与其前一层的所有神经元进行全连接（1024*784）．全连接层可以整合卷积层或者池化层中具有类别区分性的局部信息
# 全连接就是把以前的局部特征重新通过权值矩阵组装成完整的图
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
# 隐藏层-2 1024*128
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# 隐藏层-2 128*128
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# 输出层 128*10，创建10个神经元的全连接层传递给输出 ，手写数字0-9总共10个数，
# 一般采用softmax逻辑回归（softmax regression）进行分类，将输出映射为0~9的10个类别
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# 全连接层是否越多越好呢？
# 全连接层神经元越多，模型越大，速度越慢，模型拟合能力越高，在训练集上准确率越高，但过拟合风险也越大。
# 反之，全连接层神经元越少，模型越小，速度越快，模型拟合能力越低，在训练集上准确率越低，但训练集与测试集上的差距越小

# summary输出模型各层的参数状况
# 能看到数据经过每个层后输出的数据维度 以及 Param每层参数个数
model.summary()

# 设置优化器、损失函数
# optimizer：sgd（随机梯度下降）、adagrad（自适应梯度）、adadelta（adagrad的扩展）、adam（adagrad的优化）
# loss：mse（均方损失函数）、sparse_categorical_crossentropy（交叉熵损失函数）
# metrics：accuracy、sparse_accuracy、sparse_categorical_accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型，3轮
# epochs：所有训练数据集都训练过一次
# batch_size：在训练集中选择一组样本用来更新权值。1个batch包含的样本的数目，通常设为2的n次幂，网络较小时选用256，较大时选用64（每一批次数据多少）
# iteration：完成一次epoch所需的batch个数（批次数）
model.fit(x_train, y_train, epochs=3)

# 评估模型
val_loss, val_acc = model.evaluate(x_test, y_test)
print("测试集损失率：", val_loss)
print("测试集准确率：", val_acc)

# 预测模型
predictions = model.predict(x_test)
# 查看一下预测数据集的前25个
# argmax：返回索引最大值
# axis：对于二维向量而言，0代表对行进行最大值选取，此时对每一列进行操作；1代表对列进行最大值选取，此时对每一行进行操作
pred = np.argmax(predictions[:25, :], axis=1)

# 用图形展示一下跟实际值对比
plt.figure(figsize=(10, 10))
for i in range(25):
    # 绘制子图：5行、5列、位置
    plt.subplot(5, 5, i+1)
    # 不显示横坐标
    plt.xticks([])
    # 不显示纵坐标
    plt.yticks([])
    # 关闭背景的网格线
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel("true: {} / pred: {}".format(y_test[i], pred[i]))
plt.show()
