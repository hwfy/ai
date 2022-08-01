import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

# num_words: 定义字典中单词的数量，所有高于10000的单词编号都将被2（unknown）代替
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# 标签中只有 0 和 1
# 将评论情绪二元分类，评分<5为负面，标签为0，评分>7为积极评论，标签为1
# print(train_labels[:10])
# print(test_labels[:10])

# 一个映射单词到整数索引的词典，返回类型为字典
word_index = imdb.get_word_index()

# 保留第一个索引
# 因为word_index编号是从1开始的，下面添加了4个单词占用了0、1、2、3，因此v+3就从4开始，因此保留第一个索引
# 解释：https://coding.imooc.com/learn/questiondetail/176929.html
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 将key和value的顺序换一下，这样才好根据编号查单词
# dict可以转换数组元组为字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 根据单词编号获取单词
def decode_review(text):
    # reverse_word_index[i]，当字典中不存在该键时会返回KeyError类型错误
    # reverse_word_index.get(i, '?')，不存在键时会返回一个None，这里设置不存在时返回 ?
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 查看第1条评论
# train_data：(25000, n)，总共25000条评论，每条评论n个单词
words = decode_review(train_data[0])
print("第一条评论：", words)


# 将序列填充到相同的长度，原本train_data=(25000,)，填充后(25000, 256)
# value：需要填充的值，word_index["<PAD>"]对应0，默认0
# padding：pre为在序列前进行拉伸或者截断，post是在序列最后进行拉伸或者截断
# maxlen：代表所有序列的最大长度，如果没有提供默认为补齐到最长序列
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

# 可以看到每一列长度都是256，最后不足的用0补充
print("每条评论填充后长度：", len(train_data[0]), len(train_data[1]))

# keras只能接受长度相同的序列输入，因此需要填充
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)


model = keras.Sequential()
# 深度学习模型在处理文本数据时不会直接把原始文本数据作为输入，它只能处理数值张量，因此使用Embedding
# 将正整数(索引)转换为固定大小的密集向量
# 其中1000表示input_dimension,也就是词汇表的大小,每个词被映射为16维向量，10000*16
# 详细介绍：https://www.jianshu.com/p/a3f3033a7379/
model.add(keras.layers.Embedding(10000, 16))
# 对输入中的单个句子的长度进行平均池化
# 通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。这样，模型便能够以尽可能简单的方式处理各种长度的输入
model.add(keras.layers.GlobalAveragePooling1D())
# 创建具有16个神经元的全连接层，激活函数relu
model.add(keras.layers.Dense(16, activation='relu'))
# rele公式：max(0, x)，>=0
model.add(keras.layers.Dense(16, activation='relu'))
# sigmoid：归一化是把输出值压缩到0-1范围之内，把量纲转为无量纲的过程，提升模型的收敛速度
# sigmoid公式：1/(1+e-x次方)，0-1之间
model.add(keras.layers.Dense(1, activation='sigmoid'))
# 打印概述
print(model.summary())

# 设置优化器、损失函数（binary_crossentropy经常搭配sigmoid）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 将训练集拆分成两部分，用后15000个数据进行训练，用前10000个数据进行优化从而提高泛化能力，防止神经网络过度的“模仿”训练集数据，
# 不能将测试集的数据用来提升泛化能力，泛化是处理未处理过的数据，用测试集优化网络话无法对训练好的神经网络进行有效的评价
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# validation_data：在每个epoch之后,或者每⼏个epoch,验证⼀次验证集,⽤来及早发现过拟合、超参数等设置问题，⽅便我们及时调整
# verbose：0-不在标准输出流输出日志信息、1-输出进度条记录、2-每个epoch输出一行记录，默认1
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,  # 遍历40轮
                    batch_size=512,  # 每一轮512
                    validation_data=(x_val, y_val),
                    verbose=1)

# 模型评估
# verbose：0 - 不在标准输出流输出日志信息、 1 - 输出进度条记录，默认1
results = model.evaluate(test_data, test_labels, verbose=1)
print("测试集损失率: {}".format(results[0]))
print("测试集准确率：{}".format(results[1]))

# 输出训练记录
history_dict = history.history
# accuracy：历史训练准确率，类型list，长度40（训练40轮）
acc = history_dict['accuracy']
# val_accuracy：历史验证准确率，类型list，长度40（训练40轮）
val_acc = history_dict['val_accuracy']
# loss：历史训练误差，类型list，长度40（训练40轮）
loss = history_dict['loss']
# val_loss：历史验证误差，类型list，长度40（训练40轮）
val_loss = history_dict['val_loss']

# 产生数字1-40的数组
epochs = range(1, len(acc) + 1)
# b为蓝色，o为圆点
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 显示小图例，也就是plot中label的输出图例
plt.legend()
plt.show()
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 测试集预测：
n = 10
prediction = model.predict(test_data[:n])
# 将10行1列，转换为1行10列，二维转换一维
print(prediction.reshape(n))
# np.round取整（会四舍五入）：e-03 = 10的-3次方 = 0.001； e-01 = 10的-1次方 = 0.1
# 取整以后数组里 只有0/1
print(np.round(prediction.reshape(n)))
# 真实标签 只有0/1
print(test_labels[:n])
# 最后查看相应位置匹配结果（0是否匹配0，1是否匹配1）

