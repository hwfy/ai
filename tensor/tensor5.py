import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# 将评论情绪二元分类，评分<5为负面，标签为0，评分>7为积极评论，标签为1
print(train_labels[:10])
print(train_data[0])
print(len(train_data[0]))

print(train_data[1])
print(len(train_data[1]))

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


words = decode_review(train_data[0])
print(words)

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# 输入形状是用于电影评论的词汇数目
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#  18ms/step - loss: 0.0936 - accuracy: 0.9758 - val_loss: 0.3104 - val_accuracy: 0.8825
#  19ms/step - loss: 0.0168 - accuracy: 0.9985 - val_loss: 0.5547 - val_accuracy: 0.8680
#  20ms/step - loss: 0.0041 - accuracy: 0.9996 - val_loss: 0.7668 - val_accuracy: 0.8632

print(model.summary())

# # 设置优化器、损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,  # 遍历40
                    batch_size=512,  # 每一轮512
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

print("测试集准确率：{}".format(results[1]))

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

n = 10
prediction = model.predict(test_data[:n])
print(prediction.reshape(n))

print(np.round(prediction.reshape(n)))
print(test_labels[:n])


