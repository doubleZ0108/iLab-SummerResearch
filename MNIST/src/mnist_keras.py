import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils    # 使用one-hot 编码将输出标签的向量转化为布尔矩阵
from keras.layers.core import Dense, Dropout, Activation
'''
Dense: 全连接层
Dropout: 在训练过程中每次更新参数时按一定概率随机断开输入神经元, 防止过拟合
'''

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()   # 加载数据
                                                                            # 第一个tuple存储已经人工分类好的图片(每一个菇片都有自己对应的标签)
                                                                            # 第二个tuple存储没分类的图片
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
    
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')    # uint不能有负数, 要先转换为float类型
X_test = X_test.astype('float32')

# 给定的像素灰度是0~255, 为了使模型的巡训练效果更高, 通常将数值归一化映射到0-1
X_train /= 255
X_test /= 255
'''
可以以使用其他方法进行归一化
X_train = (X_train - 127) /127
X_test = (X_test - 127) / 127
'''

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 建立一个Sequential模型, 然后一层层地加入神经元
model = Sequential()

'''
Dense层: 全连接层
所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加
units -> 512 代表该层的输出维度
'''
model.add(Dense(512, input_shape=(784,)))   # 第一层要指定数据规模

'''
Activation层: 激活层
activation -> relu 将要使用的激活函数
'''
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
        
'''
Dropout层: 
rate -> 0.2 在训练过程中每次更新参数时按此概率随机断开输入神经元, 防止过拟合
'''
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data


model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(10))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.
        
'''
损失函数loss: 模型试图最小化的目标函数设为categorical_crossentropy
优化器optimizer: 使用adam算法
指标列表metrics
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
:param x: 如果模型只有一个输入, 那x应为numpy array, 如果模型有多个输入, 那x的类型应为list, list的元素是对应于各个输入的numpy array
:param y: 同x
:param batch_size: 指定进行梯度下降时每个batch包含的样本数; 训练时一个batch的样本会呗计算一次梯度下降, 使目标函数优化一步
:param epochs: 训练达到epoch值时停止
:param verbose: 日志显示(0=>不在标准输出流输出日志信息 1=>输出进度条记录 2=>每个epoch输出一行记录
:param validation_data: 验证集
'''
model.fit(X_train, Y_train,
          batch_size=128, epochs=4,verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
