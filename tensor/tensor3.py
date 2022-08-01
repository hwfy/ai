import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# 5000张图片
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print("training entries: {}, labels: {}".format(len(train_images), len(train_labels)))

# 总共10类
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 显示训练集图片：
# 填充画布大小
plt.figure(figsize=(10, 10))
for i in range(25):
    # 绘制小图，5行5列，从第1个位置开始
    plt.subplot(5, 5, i+1)
    # 不显示横坐标
    plt.xticks([])
    # 不显示纵坐标
    plt.yticks([])
    # 关闭背景网格线
    plt.grid(False)
    # 显示训练集图片
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 下方文字描述
    # CIFAR的标签是数字，通过映射取得英文名。
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
plt.clf()

# 归一化：训练集和测试集中图像的像素值处于0-255之间，将这些值缩小到0-1之间
# 如果不归一化，尺度大的特征值，梯度也比较大，尺度小的特征值，梯度也比较小，
# 而梯度更新时的学习率是一样的，如果学习率小，梯度小的就更新慢，如果学习率大，梯度大的方向不稳定，不易收敛，
# 通常需要使用最小的学习率迁就大尺度的维度才能保证损失函数有效下降，
# 因此，通过归一化，把不同维度的特征值范围调整到相近的范围内，就能统一使用较大的学习率加速学习
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
# 创建卷积层：32个卷积核，卷积核尺寸(3高*3宽*3深)，输入图像大小(32长*32宽*3通道（RGB）)，黑白为1
# 输入通道数为3（32*32*3），决定了卷积核深度也是3（3*3*3）
# 输出深度和卷积核数相关，一个卷积核对应一层输出
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 创建最大池化层：池化核尺寸(2*2)，默认2*2
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# 卷积层后不一定要加池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 将输入层的数据压成一维的数据，一般用在卷积层和全连接层之间（因为全连接层只能接收一维数据，而卷积层可以处理二维数据
model.add(layers.Flatten())
# 创建全连接层，16个神经元
model.add(layers.Dense(64, activation='relu'))
# 创建输出层，因为CIFAR数据集有10类，所以需要10个输出
model.add(layers.Dense(10))
# 输出模型各层的参数
print(model.summary())


# 设置优化器、损失函数
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# SparseCategoricalCrossentropy 函数用于计算多分类问题的交叉熵
# from_logits=True 表示输出层是不带softmax激活函数的，设置True可以更好解决softmax溢出问题，默认为False
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 模型训练：
# validation_data：在每个epoch之后或者每⼏个epoch,验证⼀次验证集,⽤来及早发现过拟合、超参数等设置问题，⽅便我们及时调整
# verbose：0-不在标准输出流输出日志信息、1-输出进度条记录、2-每个epoch输出一行记录，默认1
history = model.fit(train_images,
                    train_labels,
                    # batch_size=100,
                    epochs=10,
                    # verbose=1,
                    validation_data=(test_images, test_labels)
                    # validation_split=0.1
                    )

# epochs 10次
# 24ms/step - loss: 0.6014 - accuracy: 0.7892 - val_loss: 0.8766 - val_accuracy: 0.7088
# epochs 15次
# 24ms/step - loss: 0.4239 - accuracy: 0.8489 - val_loss: 1.0174 - val_accuracy: 0.6994

# 展示历史测试准确率和验证准确率
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], label='accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# 设置y轴最小和最大值，xlim设置x轴最小和最大值，如果不设置曲线可能会溢出区域
plt.ylim([0.5, 1])
# 显示小图例，也就是plot中label的输出图例
# loc：小图例位置，lower下方、right右边
plt.legend(loc='lower right')
plt.show()
plt.clf()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# 和val_loss差不多
print("测试集损失率: ", test_loss)
# 和val_accuracy差不多
print("测试集精确度: ", test_acc)



