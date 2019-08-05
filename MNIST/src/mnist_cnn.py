from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# 第一层: 卷积层, 有32个滤波器, 卷积核大小为3*3, 32个, 第一层要输入训练图片的规模
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 第二层: 卷积层, 卷积核大小为3*3 64个
model.add(Conv2D(64, (3, 3), activation='relu'))

# 第三层: 池化层, 使用MaxPooling, 大小为2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第四层: Dropout层, 对参数进行正则化防止模型过拟合
model.add(Dropout(0.25))

# 第五层: 将三维张亮转换为一维向量, 展开前张亮的维度是(12,12,64), 转化为一维(9216)
model.add(Flatten())

# 使用Dense构建了2层全相连层, 逐步将一位向量的位数从9216变为128, 最终变为10
# 第六层: 全向量层, 有128个神经元, 激活函数采用‘relu’
model.add(Dense(128, activation='relu'))
# 第七层: 训练过程中每次更新参数时随机断开输入神经元
model.add(Dropout(0.5))
# 第八层: 激活函数为softmax, 10位恰好对应0~9十个数字
model.add(Dense(num_classes, activation='softmax'))

# 打印定义的模型结构
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
