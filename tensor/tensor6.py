import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print("training entries: {}, labels: {}".format(len(train_images), len(train_labels)))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # CIFAR 的标签是 array，需要额外的索引。
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
plt.clf()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

# 24ms/step - loss: -16.5369 - accuracy: 0.1005 - val_loss: -16.5370 - val_accuracy: 0.1009
# 22ms/step - loss: 0.6462 - accuracy: 0.7728 - val_loss: 0.8569 - val_accuracy: 0.7143
# 25ms/step - loss: 0.6267 - accuracy: 0.7813 - val_loss: 0.8605 - val_accuracy: 0.7085
# 15次
# 24ms/step - loss: 0.4239 - accuracy: 0.8489 - val_loss: 1.0174 - val_accuracy: 0.6994


# 设置优化器、损失函数
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images,
                    train_labels,
                    # batch_size=100,
                    epochs=3,
                    # verbose=1,
                    validation_data=(test_images, test_labels))
                    # validation_split=0.1)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.clf()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("精确度: ", test_acc)



