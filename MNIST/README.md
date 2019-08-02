# MINST手写数字识别

- [MINST手写数字识别](#minst------)
  * [minst数据集](#minst---)
    + [图片](#--)
    + [标签](#--)
    + [one-hot编码(独热编码)](#one-hot--------)
  * [Keras神经网络](#keras----)
    + [输入/输出/标签](#--------)
    + [损失函数 -> 交叉熵](#-----------)
    + [回归模型](#----)
    + [学习速率](#----)
    + [激活函数 -> softmax](#--------softmax)
    + [具体实现](#----)
      - [下载该数据集](#------)
      - [导入依赖](#----)
      - [装载训练数据](#------)
        * [查看训练集中的例子](#---------)
      - [格式化训练数据](#-------)
      - [搭建神经网络](#------)
        * [搭建三层全相连网络](#---------)
        * [编译模型](#----)
        * [训练模型](#----)
        * [性能评估](#----)
      - [检查输出](#----)


## minst数据集

**该数据集包含以下四个部分**

- train-images-idx3-ubyte.gz: 训练集-图片，6w
- train-labels-idx1-ubyte.gz: 训练集-标签，6w
- t10k-images-idx3-ubyte.gz: 测试集-图片，1w
- t10k-labels-idx1-ubyte.gz: 测试集-标签，1w

### 图片

每张图片大小为28*28像素, 可以用28\*28大小的数组来表示一张图片

### 标签

用大小为10的数组来表示

### one-hot编码(独热编码)

使用N位表示N种状态, 任何时候只有其中的一位有效

> 例如
>
> 性别:  
> [0, 1]代表女，[1, 0]代表男
>
> 数字0-9: 
> [0,0,0,0,0,0,0,0,0,1]代表9，[0,1,0,0,0,0,0,0,0,0]代表1

**优点:**

- 能够处理非连续性数值特征
- 一定程度上扩充了特征(性别本身是一个特征, 经过编码以后, 就变成了男或女两个特征)
- 容错性: 比如神经网络的输出结果是 [0,0.1,0.2,0.7,0,0,0,0,0, 0]转成独热编码后，表示数字3。即值最大的地方变为1，其余均为0。[0,0.1,0.4,0.5,0,0,0,0,0, 0]也能表示数字3。

## Keras神经网络

### 输入/输出/标签

- 输入: 传入给网络处理的向量, *784(28*28)的向量*
- 输出: 网络处理后返回的结果: *大小为10的概率向量*
- 标签: 期望网络返回的结果

### 损失函数 -> 交叉熵

**作用**: 评估网络模型的好坏

**目标:** 传入大量的训练集训练目标, 将损失函数的值降到最小

 **函数形式:** `-sum(label * log(y))`

**优点**: 只关注独热编码中有效位的损失, 屏蔽了无效位的变化(不会影响最终的结果), 并且通过取对数放大了有效位的损失

> 例: [0, 0, 1] 与 [0.1, 0.3, 0.6]的交叉熵为 -log(0.6) = 0.51[0, 0, 1] 与 [0.2, 0.2, 0.6]的交叉熵为 -log(0.6) = 0.51[0, 0, 1] 与 [0.1, 0, 0.9]的交叉熵为 -log(0.9) = 0.10

![image.png](https://upload-images.jianshu.io/upload_images/12014150-2f418be5d976cf21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 回归模型

**作用**: 如果把网络理解为一个函数, 回归模型是希望对这个函数进行拟合

**方法**: 不断地传入X和label的值, 来修正w和b, 使得最终得到的Y和label的loss最小

可以采用梯度下降法, 找到最快的方向, 调整w和b的值, 使得w*X + b的值越来越接近label

### 学习速率

![image.png](https://upload-images.jianshu.io/upload_images/12014150-02bf061551a313c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 激活函数 -> softmax

再计算交叉熵前的Y值是经过softmax后的，经过softmax后的Y，并不影响Y向量的每个位置的值之间的大小关系。但是有如下2个作用:

- 一是放大效果
- 二是梯度下降时需要一个可导的函数。

<u>softmax函数将任意n维的实值向量转换为取值范围在(0,1)之间的n维实值向量，并且总和为1。</u>

> 例如：向量softmax([1.0, 2.0, 3.0]) ------> [0.09003057, 0.24472847, 0.66524096]

**性质：**

1. 因为softmax是单调递增函数，因此不改变原始数据的大小顺序。
2. 将原始输入映射到(0,1)区间，并且总和为1，常用于表征概率。
3. softmax(x) = softmax(x+c), 这个性质用于保证数值的稳定性。

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

softmax([4, 5, 10])
# [ 0.002,  0.007,  0.991]
```

### 具体实现

#### 下载该数据集

- 在官网上下载`mnist.npz`数据集
- 将其放于`~/.keras/datasets/mnist.npz`中

#### 导入依赖

```python
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
```

```
Using TensorFlow backend.
```

#### 装载训练数据

通过keras自带的数据集mnist导入数据, 对其进行归一化处理, 并将原而为数据变成一位数据, 作为网络的输入

```python
nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()   # 加载数据
                                                                            # 第一个tuple存储已经人工分类好的图片(每一个菇片都有自己对应的标签)
                                                                            # 第二个tuple存储没分类的图片
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
```

```
X_train original shape (60000, 28, 28)
y_train original shape (60000,)
```

X_train是list类型的对象, 存储的是28*28的图像像素   
Y_train存储的是图像对应的标签(也就是该图片代表什么数字)

##### 查看训练集中的例子

```python
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
```

![output_6_0.png](https://upload-images.jianshu.io/upload_images/12014150-eba549ecc2c04684.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 格式化训练数据

对于每一个训练样本, 我们的神经网络的到单个的数组

- 先将28*28的图片变形成784长度的向量

```python
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
```

```
Training matrix shape (60000, 784)
Testing matrix shape (10000, 784)
```

- one-hot 编码: 再将输入从[0,255]压缩到[0,1]

> 0 -> [1,0,0,0,0,0,0,0,0]
>
> 1 -> [0,1,0,0,0,0,0,0,0]
>
> 2 -> [0,0,1,0,0,0,0,0,0]
>
> etc.

```python
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
```

#### 搭建神经网络

##### 搭建三层全相连网络

<img src="https://upload-images.jianshu.io/upload_images/12014150-db9f8bbfa9a48a3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"/>

```python
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
```

##### 编译模型

训练模型之前, 需要通过编译对学习过程进行配置

- 损失函数: 分类交叉熵(用于比较两个概率分布函数)
  - 预测是个不同数字的概率分布, 目标是一个概率分布, 正确类别为100%, 其他所有类别为0
  - 例如, 80%认为这个图片是3, 10%认为是2, 5%认为是1等
- 优化器: 帮助模型快速的学习, 同时防止“卡住”和“爆炸”情况

```python
'''
损失函数loss: 模型试图最小化的目标函数设为categorical_crossentropy
优化器optimizer: 使用adam算法
指标列表metrics
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

##### 训练模型

```python
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
```

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/4
60000/60000 [==============================] - 3s 45us/step - loss: 0.2525 - acc: 0.9235 - val_loss: 0.1099 - val_acc: 0.9657
Epoch 2/4
60000/60000 [==============================] - 2s 39us/step - loss: 0.1028 - acc: 0.9686 - val_loss: 0.0735 - val_acc: 0.9770
Epoch 3/4
60000/60000 [==============================] - 2s 41us/step - loss: 0.0704 - acc: 0.9782 - val_loss: 0.0666 - val_acc: 0.9786
Epoch 4/4
60000/60000 [==============================] - 3s 52us/step - loss: 0.0573 - acc: 0.9814 - val_loss: 0.0798 - val_acc: 0.9746
```

##### 性能评估

```python
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
```

```
Test score: 0.07981417043644469
Test accuracy: 0.9746
```

#### 检查输出

- 正确的例子
- 错误的例子

```python
# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
```

```python
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
```

![output_21_0.png](https://upload-images.jianshu.io/upload_images/12014150-5917bdb1a0aedadb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![output_21_1.png](https://upload-images.jianshu.io/upload_images/12014150-7dc3389aa39b7af8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)







在第一个tuple训练完后，我们可以把第二个tuple利用神经网络进行分类，根据实验结果的真实值与我们的预测值进行对比得到相应的损失值，再利用反向传播进行参数更新，再进行分类，然后重复前述步骤直至损失值最小




![image-20190731194915115.png](https://upload-images.jianshu.io/upload_images/12014150-3f302abe4e10ffdc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

