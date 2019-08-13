# 吴恩达课后编程作业


Table of Contents
=================

   * [吴恩达课后编程作业](#吴恩达课后编程作业)
      * [【吴恩达课后编程作业】Course 1 - 神经网络和深度学习 - 第四周 - PA1&amp;2 - 一步步搭建多层神经网络以及应用](#吴恩达课后编程作业course-1---神经网络和深度学习---第四周---pa12---一步步搭建多层神经网络以及应用)
         * [开始之前](#开始之前)
         * [准备软件包](#准备软件包)
         * [初始化参数](#初始化参数)
         * [前向传播函数](#前向传播函数)
            * [线性部分【LINEAR】](#线性部分linear)
            * [线性激活部分【LINEAR - &gt;ACTIVATION】](#线性激活部分linear---activation)
         * [计算成本](#计算成本)
         * [反向传播](#反向传播)
            * [线性部分【LINEAR backward】](#线性部分linear-backward)
            * [线性激活部分【LINEAR -&gt; ACTIVATION backward】](#线性激活部分linear---activation-backward)
         * [更新参数](#更新参数)
         * [搭建两层神经网络](#搭建两层神经网络)
         * [搭建多层神经网络](#搭建多层神经网络)
         * [分析](#分析)
         * [相关库代码](#相关库代码)
            * [lr_utils.py](#lr_utilspy)
            * [dnn_utils.py](#dnn_utilspy)
            * [testCase.py](#testcasepy)
      * [【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第一周作业(1&amp;2&amp;3) - 初始化、正则化、梯度校验](#吴恩达课后编程作业course-2---改善深层神经网络---第一周作业123---初始化正则化梯度校验)
         * [开始之前](#开始之前-1)
         * [初始化参数](#初始化参数-1)
            * [读取并绘制数据](#读取并绘制数据)
            * [初始化为零](#初始化为零)
            * [随机初始化](#随机初始化)
            * [抑梯度异常初始化](#抑梯度异常初始化)
         * [正则化模型](#正则化模型)
            * [读取并绘制数据集](#读取并绘制数据集)
            * [不使用正则化](#不使用正则化)
            * [使用正则化](#使用正则化)
               * [L2正则化](#l2正则化)
            * [随机删除节点](#随机删除节点)
         * [梯度校验](#梯度校验)
            * [一维线性](#一维线性)
            * [高维](#高维)
         * [相关库代码](#相关库代码-1)
            * [init_utils.py](#init_utilspy)
            * [reg_utils.py](#reg_utilspy)
            * [gc_utils.py](#gc_utilspy)
      * [【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第二周作业 - 优化算法](#吴恩达课后编程作业course-2---改善深层神经网络---第二周作业---优化算法)
         * [开始之前](#开始之前-2)
         * [导入库函数](#导入库函数)
         * [梯度下降](#梯度下降)
         * [mini-batch梯度下降法](#mini-batch梯度下降法)
         * [包含动量的梯度下降](#包含动量的梯度下降)
         * [Adam算法](#adam算法)
         * [测试](#测试)
            * [加载数据集](#加载数据集)
            * [定义模型](#定义模型)
            * [梯度下降测试](#梯度下降测试)
            * [具有动量的梯度下降测试](#具有动量的梯度下降测试)
            * [Adam优化后的梯度下降](#adam优化后的梯度下降)
         * [总结](#总结)
         * [相关库代码](#相关库代码-2)
            * [opt_utils.py](#opt_utilspy)
            * [testCase.py](#testcasepy-1)
      * [【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第三周作业 - TensorFlow入门](#吴恩达课后编程作业course-2---改善深层神经网络---第三周作业---tensorflow入门)
         * [TensorFlow 入门](#tensorflow-入门)
            * [1 - 导入TensorFlow库](#1---导入tensorflow库)
            * [1.1 - 线性函数](#11---线性函数)
            * [1.2 - 计算sigmoid](#12---计算sigmoid)
            * [1.3 - 计算成本](#13---计算成本)
            * [1.4 - 使用独热编码（0、1编码）](#14---使用独热编码01编码)
            * [1.5 - 初始化为0和1](#15---初始化为0和1)
            * [2 - 使用TensorFlow构建你的第一个神经网络](#2---使用tensorflow构建你的第一个神经网络)
            * [2.0 - 要解决的问题](#20---要解决的问题)
            * [2.1 - 创建placeholders](#21---创建placeholders)
            * [2.2 - 初始化参数](#22---初始化参数)
            * [2.3 - 前向传播](#23---前向传播)
            * [2.4 - 计算成本](#24---计算成本)
            * [2.5 - 反向传播&amp;更新参数](#25---反向传播更新参数)
            * [2.6 - 构建模型](#26---构建模型)
         * [相关库代码](#相关库代码-3)
            * [tf_utils.py](#tf_utilspy)





## 【吴恩达课后编程作业】Course 1 - 神经网络和深度学习 - 第四周 - PA1&2 - 一步步搭建多层神经网络以及应用

### 开始之前

我们要构建两个神经网络，一个是构建两层的神经网络，一个是构建多层的神经网络，多层神经网络的层数可以自己定义。在这里，我们简单的讲一下难点，本文会提到[LINEAR-> ACTIVATION]转发函数，比如我有一个多层的神经网络，结构是输入层->隐藏层->隐藏层->···->隐藏层->输出层，在每一层中，我会首先计算Z = np.dot(W,A) + b，这叫做【linear_forward】，然后再计算A = relu(Z) 或者 A = sigmoid(Z)，这叫做【linear_activation_forward】，合并起来就是这一层的计算方法，所以每一层的计算都有两个步骤，先是计算Z，再计算A，你也可以参照下图： ![GALBOL](https://img-blog.csdn.net/20180331114317342?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们来说一下步骤：

​	1. 初始化网络参数

​	2. 前向传播

​		2.1 计算一层的中线性求和的部分

​		2.2 计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）

​		2.3 结合线性求和与激活函数

​	3. 计算误差

​	4. 反向传播

​		4.1 线性部分的反向传播公式

​		4.2 激活函数部分的反向传播公式

​		4.3 结合线性部分与激活函数的反向传播公式

​	5. 更新参数

​		请注意，对于每个前向函数，都有一个相应的后向函数。 这就是为什么在我们的转发模块的每一步都会在cache中存储一些值，cache的值对计算梯度很有用， 在反向传播模块中，我们将使用cache来计算梯度。 现在我们正式开始分别构建两层神经网络和多层神经网络。

### 准备软件包

在开始我们需要准备一些软件包：

```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases #参见资料包，或者在文章底部copy
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import lr_utils #参见资料包，或者在文章底部copy
```

软件包准备好了，我们开始构建初始化参数的函数。

为了和我的数据匹配，你需要指定随机种子

```python
np.random.seed(1)
```

### 初始化参数

对于一个两层的神经网络结构而言，模型结构是线性->ReLU->线性->sigmod函数。

初始化函数如下：

```python
def initialize_parameters(n_x,n_h,n_y):
    """
    此函数是为了初始化两层网络参数而使用的函数。
    参数：
        n_x - 输入层节点数量
        n_h - 隐藏层节点数量
        n_y - 输出层节点数量

返回：
    parameters - 包含你的参数的python字典：
        W1 - 权重矩阵,维度为（n_h，n_x）
        b1 - 偏向量，维度为（n_h，1）
        W2 - 权重矩阵，维度为（n_y，n_h）
        b2 - 偏向量，维度为（n_y，1）

"""
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))

# 使用断言确保我的数据格式是正确的

assert(W1.shape == (n_h, n_x))
assert(b1.shape == (n_h, 1))
assert(W2.shape == (n_y, n_h))
assert(b2.shape == (n_y, 1))

parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

return parameters  
```

初始化完成我们来测试一下：

```python
print("==============测试initialize_parameters==============")
parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
==============测试initialize_parameters==============
W1 = [[ 0.01624345 -0.00611756 -0.00528172]
 [-0.01072969  0.00865408 -0.02301539]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[ 0.01744812 -0.00761207]]
b2 = [[ 0.]]
```

两层的神经网络测试已经完毕了，那么对于一个L层的神经网络而言呢？初始化会是什么样的？

在实际中，我们来看一下它是怎样计算的吧：

```python
def initialize_parameters_deep(layers_dims):
    """
    此函数是为了初始化多层网络参数而使用的函数。
    参数：
        layers_dims - 包含我们网络中每个图层的节点数量的列表

返回：
    parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                 W1 - 权重矩阵，维度为（layers_dims [1]，layers_dims [1-1]）
                 bl - 偏向量，维度为（layers_dims [1]，1）
"""
np.random.seed(3)
parameters = {}
L = len(layers_dims)

for l in range(1,L):
    parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
    parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

# 确保我要的数据的格式是正确的
	assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
  assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

return parameters
```

测试一下：

```python
# 测试initialize_parameters_deep

print("==============测试initialize_parameters_deep==============")
layers_dims = [5,4,3]
parameters = initialize_parameters_deep(layers_dims)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
==============测试initialize_parameters_deep==============
W1 = [[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
 [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
 [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
 [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.01185047 -0.0020565   0.01486148  0.00236716]
 [-0.01023785 -0.00712993  0.00625245 -0.00160513]
 [-0.00768836 -0.00230031  0.00745056  0.01976111]]
b2 = [[ 0.]
 [ 0.]
 [ 0.]]
```

我们分别构建了两层和多层神经网络的初始化参数的函数，现在我们开始构建前向传播函数。

### 前向传播函数

前向传播有以下三个步骤

- LINEAR
- LINEAR - >ACTIVATION，其中激活函数将会使用ReLU或Sigmoid。
- [LINEAR - > RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）

#### 线性部分【LINEAR】

前向传播中，线性部分计算如下：

```python
def linear_forward(A,W,b):
    """
    实现前向传播的线性部分。

参数：
    A - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
    W - 权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
    b - 偏向量，numpy向量，维度为（当前图层节点数量，1）

返回：
     Z - 激活功能的输入，也称为预激活参数
     cache - 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
"""
Z = np.dot(W,A) + b
assert(Z.shape == (W.shape[0],A.shape[1]))
cache = (A,W,b)

return Z,cache
```

测试一下线性部分：

```python
# 测试linear_forward

print("==============测试linear_forward==============")
A,W,b = testCases.linear_forward_test_case()
Z,linear_cache = linear_forward(A,W,b)
print("Z = " + str(Z))
```

测试结果：

```python
==============测试linear_forward==============
Z = [[ 3.26295337 -1.23429987]]
```

我们前向传播的单层计算完成了一半了，我们来开始构建后半部分。

#### 线性激活部分【LINEAR - >ACTIVATION】

为了更方便，我们将把两个功能（线性和激活）分组为一个功能（LINEAR-> ACTIVATION）。 因此，我们将实现一个执行LINEAR前进步骤，然后执行ACTIVATION前进步骤的功能。

我们为了实现LINEAR->ACTIVATION这个步骤， 使用的公式是：A[l]=g(Z[l])=g(W[l]A[l−1]+b[l])，其中，函数g会是sigmoid() 或者是 relu()，当然，sigmoid()只在输出层使用,现在我们正式构建前向线性激活部分。
```python
def linear_activation_forward(A_prev,W,b,activation):
    """
    实现LINEAR-> ACTIVATION 这一层的前向传播

参数：
    A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
    W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
    b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）
    activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】

返回：
    A - 激活函数的输出，也称为激活后的值
    cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
"""

if activation == "sigmoid":
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)
elif activation == "relu":
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = relu(Z)

assert(A.shape == (W.shape[0],A_prev.shape[1]))
cache = (linear_cache,activation_cache)

return A,cache
```

测试一下：

```python
# 测试linear_activation_forward

print("==============测试linear_activation_forward==============")
A_prev, W,b = testCases.linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("sigmoid，A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("ReLU，A = " + str(A))
```

测试结果：

```python
==============测试linear_activation_forward==============
sigmoid，A = [[ 0.96890023  0.11013289]]
ReLU，A = [[ 3.43896131  0.        ]]
```

我们把两层模型需要的前向传播函数做完了，那多层网络模型的前向传播是怎样的呢？我们调用上面的那两个函数来实现它，为了在实现L层神经网络时更加方便，我们需要一个函数来复制前一个函数（带有RELU的linear_activation_forward）L-1次，然后用一个带有SIGMOID的linear_activation_forward跟踪它，我们来看一下它的结构是怎样的：

![[LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID model](https://img-blog.csdn.net/20180331134637680?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在下面的代码中，AL表示A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L]). (也可称作 Yhat,数学表示为 Y^.)

多层模型的前向传播计算模型代码如下：

```python
def L_model_forward(X,parameters):
    """
    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION

参数：
    X - 数据，numpy数组，维度为（输入节点数量，示例数）
    parameters - initialize_parameters_deep（）的输出

返回：
    AL - 最后的激活值
    caches - 包含以下内容的缓存列表：
             linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
             linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
"""
caches = []
A = X
L = len(parameters) // 2
for l in range(1,L):
    A_prev = A 
    A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
    caches.append(cache)

AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
caches.append(cache)

assert(AL.shape == (1,X.shape[1]))

return AL,caches
```

测试一下：

```python
# 测试L_model_forward

print("==============测试L_model_forward==============")
X,parameters = testCases.L_model_forward_test_case()
AL,caches = L_model_forward(X,parameters)
print("AL = " + str(AL))
print("caches 的长度为 = " + str(len(caches)))
```

测试结果：

```python
==============测试L_model_forward==============
AL = [[ 0.17007265  0.2524272 ]]
caches 的长度为 = 2
```

### 计算成本

我们已经把这两个模型的前向传播部分完成了，我们需要计算成本（误差），以确定它到底有没有在学习。

```python
def compute_cost(AL,Y):
    """
    实施等式（4）定义的成本函数。

参数：
    AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
    Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

返回：
    cost - 交叉熵成本
"""
m = Y.shape[1]
cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

cost = np.squeeze(cost)
assert(cost.shape == ())

return cost
```

测试一下：

```python
# 测试compute_cost

print("==============测试compute_cost==============")
Y,AL = testCases.compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
```

测试结果：

```python
==============测试compute_cost==============
cost = 0.414931599615
```

我们已经把误差值计算出来了，现在开始进行反向传播

### 反向传播

反向传播用于计算相对于参数的损失函数的梯度，我们来看看向前和向后传播的流程图： 

![Forward and Backward propagation](https://img-blog.csdn.net/20180331140325220?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

流程图有了，我们再来看一看对于线性的部分的公式：![Linear Pic](https://img-blog.csdn.net/20180331140510484?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

与前向传播类似，我们有需要使用三个步骤来构建反向传播：

- LINEAR 后向计算
- LINEAR -> ACTIVATION 后向计算，其中ACTIVATION 计算Relu或者Sigmoid 的结果
- [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID 后向计算 (整个模型)

#### 线性部分【LINEAR backward】

我们来实现后向传播线性部分：

```python
def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分（第L层）

参数：
     dZ - 相对于（当前第l层的）线性输出的成本梯度
     cache - 来自当前层前向传播的值的元组（A_prev，W，b）

返回：
     dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
     dW - 相对于W（当前层l）的成本梯度，与W的维度相同
     db - 相对于b（当前层l）的成本梯度，与b维度相同
"""
A_prev, W, b = cache
m = A_prev.shape[1]
dW = np.dot(dZ, A_prev.T) / m
db = np.sum(dZ, axis=1, keepdims=True) / m
dA_prev = np.dot(W.T, dZ)

assert (dA_prev.shape == A_prev.shape)
assert (dW.shape == W.shape)
assert (db.shape == b.shape)

return dA_prev, dW, db
```

测试一下：

```python
# 测试linear_backward

print("==============测试linear_backward==============")
dZ, linear_cache = testCases.linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
```

测试结果：

```python
==============测试linear_backward==============
dA_prev = [[ 0.51822968 -0.19517421]
 [-0.40506361  0.15255393]
 [ 2.37496825 -0.89445391]]
dW = [[-0.10076895  1.40685096  1.64992505]]
db = [[ 0.50629448]]
```

#### 线性激活部分【LINEAR -> ACTIVATION backward】

为了帮助实现linear_activation_backward，我们提供了两个后向函数：

- **sigmoid_backward**:实现了sigmoid（）函数的反向传播，你可以这样调用它：

```python
dZ = sigmoid_backward(dA, activation_cache)
```

- **relu_backward**: 实现了relu（）函数的反向传播，你可以这样调用它：

```python
dZ = relu_backward(dA, activation_cache)
```

我们先在正式开始实现后向线性激活：

```python
def linear_activation_backward(dA,cache,activation="relu"):
    """
    实现LINEAR-> ACTIVATION层的后向传播。

参数：
     dA - 当前层l的激活后的梯度值
     cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
     activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
返回：
     dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
     dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
     db - 相对于b（当前层l）的成本梯度值，与b的维度相同
"""
linear_cache, activation_cache = cache
if activation == "relu":
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
elif activation == "sigmoid":
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

return dA_prev,dW,db
```

测试一下：

```python
# 测试linear_activation_backward

print("==============测试linear_activation_backward==============")
AL, linear_activation_cache = testCases.linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
```

测试结果：

```python
==============测试linear_activation_backward==============
sigmoid:
dA_prev = [[ 0.11017994  0.01105339]
 [ 0.09466817  0.00949723]
 [-0.05743092 -0.00576154]]
dW = [[ 0.10266786  0.09778551 -0.01968084]]
db = [[-0.05729622]]

relu:
dA_prev = [[ 0.44090989 -0.        ]
 [ 0.37883606 -0.        ]
 [-0.2298228   0.        ]]
dW = [[ 0.44513824  0.37371418 -0.10478989]]
db = [[-0.20837892]]
```

我们已经把两层模型的后向计算完成了，对于多层模型我们也需要这两个函数来完成，我们来看一下流程图：![Backward pass](https://img-blog.csdn.net/20180331142737430?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在之前的前向计算中，我们存储了一些包含包含（X，W，b和z）的cache，在犯下那个船舶中，我们将会使用它们来计算梯度值，所以，在L层模型中，我们需要从L层遍历所有的隐藏层，在每一步中，我们需要使用那一层的cache值来进行反向传播。 

上面我们提到了A[L]，它属于输出层，A[L]=σ(Z[L])，所以我们需要计算dAL，我们可以使用下面的代码来计算它：

```python
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
```

 计算完了以后，我们可以使用此激活后的梯度dAL继续向后计算，我们这就开始构建多层模型向后传播函数：

```python
def L_model_backward(AL,Y,caches):
    """
    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播

    参数：
     AL - 概率向量，正向传播的输出（L_model_forward（））
     Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
     caches - 包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层
                 linear_activation_forward（"sigmoid"）的cache

    返回：
     grads - 具有梯度值的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
```

测试一下：

```python
#测试L_model_backward
print("==============测试L_model_backward==============")
AL, Y_assess, caches = testCases.L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))
```

测试结果：

```python
==============测试L_model_backward==============
dW1 = [[ 0.41010002  0.07807203  0.13798444  0.10502167]
 [ 0.          0.          0.          0.        ]
 [ 0.05283652  0.01005865  0.01777766  0.0135308 ]]
db1 = [[-0.22007063]
 [ 0.        ]
 [-0.02835349]]
dA1 = [[ 0.          0.52257901]
 [ 0.         -0.3269206 ]
 [ 0.         -0.32070404]
 [ 0.         -0.74079187]]
```

### 更新参数

我们把向前向后传播都完成了，现在我们就开始更新参数。

```python
def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数

    参数：
     parameters - 包含你的参数的字典
     grads - 包含梯度值的字典，是L_model_backward的输出

    返回：
     parameters - 包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    """
    L = len(parameters) // 2 #整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
```

测试一下：

```python
#测试update_parameters
print("==============测试update_parameters==============")
parameters, grads = testCases.update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))
```

测试结果：

```python
==============测试update_parameters==============
W1 = [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
 [-1.76569676 -0.80627147  0.51115557 -1.18258802]
 [-1.0535704  -0.86128581  0.68284052  2.20374577]]
b1 = [[-0.04659241]
 [-1.28888275]
 [ 0.53405496]]
W2 = [[-0.55569196  0.0354055   1.32964895]]
b2 = [[-0.84610769]]
```

至此为止，我们已经实现该神经网络中所有需要的函数。接下来，我们将这些方法组合在一起，构成一个神经网络类，可以方便的使用。

### 搭建两层神经网络

一个两层的神经网络模型图如下：![2-layer neural network. ](https://img-blog.csdn.net/20180331150311490?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们正式开始构建两层的神经网络:

```python
def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """
    实现一个两层的神经网络，【LINEAR->RELU】 -> 【LINEAR->SIGMOID】
    参数：
        X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_y,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱
    返回:
        parameters - 一个包含W1，b1，W2，b2的字典变量
    """
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x,n_h,n_y) = layers_dims

    """
    初始化参数
    """
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    """
    开始进行迭代
    """
    for i in range(0,num_iterations):
        #前向传播
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        #计算成本
        cost = compute_cost(A2,Y)

        #后向传播
        ##初始化后向传播
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        ##向后传播，输入：“dA2，cache2，cache1”。 输出：“dA1，dW2，db2;还有dA0（未使用），dW1，db1”。
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        ##向后传播完成后的数据保存到grads
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        #更新参数
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        #打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
    #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    #返回parameters
    return parameters
```

我们现在开始加载数据集。

```python
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
```

数据集加载完成，开始正式训练：

```python
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

parameters = two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)
```

训练结果：

```python
第 0 次迭代，成本值为： 0.69304973566
第 100 次迭代，成本值为： 0.646432095343
第 200 次迭代，成本值为： 0.632514064791
第 300 次迭代，成本值为： 0.601502492035
第 400 次迭代，成本值为： 0.560196631161
第 500 次迭代，成本值为： 0.515830477276
第 600 次迭代，成本值为： 0.475490131394
第 700 次迭代，成本值为： 0.433916315123
第 800 次迭代，成本值为： 0.40079775362
第 900 次迭代，成本值为： 0.358070501132
第 1000 次迭代，成本值为： 0.339428153837
第 1100 次迭代，成本值为： 0.30527536362
第 1200 次迭代，成本值为： 0.274913772821
第 1300 次迭代，成本值为： 0.246817682106
第 1400 次迭代，成本值为： 0.198507350375
第 1500 次迭代，成本值为： 0.174483181126
第 1600 次迭代，成本值为： 0.170807629781
第 1700 次迭代，成本值为： 0.113065245622
第 1800 次迭代，成本值为： 0.0962942684594
第 1900 次迭代，成本值为： 0.0834261795973
第 2000 次迭代，成本值为： 0.0743907870432
第 2100 次迭代，成本值为： 0.0663074813227
第 2200 次迭代，成本值为： 0.0591932950104
第 2300 次迭代，成本值为： 0.0533614034856
第 2400 次迭代，成本值为： 0.0485547856288
```

![two layers model train result](https://img-blog.csdn.net/20180331151625329?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

迭代完成之后我们就可以进行预测了，预测函数如下：

```python
def predict(X, y, parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层

    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数

    返回：
     p - 给定数据集X的预测
    """

    m = X.shape[1]
    n = len(parameters) // 2 # 神经网络的层数
    p = np.zeros((1,m))

    #根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("准确度为: "  + str(float(np.sum((p == y))/m)))

    return p
```

预测函数构建好了我们就开始预测，查看训练集和测试集的准确性：

```python
predictions_train = predict(train_x, train_y, parameters) #训练集
predictions_test = predict(test_x, test_y, parameters) #测试集
```

预测结果：

```python
准确度为: 1.0
准确度为: 0.72
```

这样看来，我的测试集的准确度要比上一次高一些，上次的是70%，这次是72%，那如果我使用更多层的圣经网络呢？

### 搭建多层神经网络

我们首先来看看多层的网络的结构![L layers neural networ](https://img-blog.csdn.net/20180331154229600?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```python
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False,isPlot=True):
    """
    实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。

    参数：
        X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱

    返回：
     parameters - 模型学习的参数。 然后他们可以用来预测。
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0,num_iterations):
        AL , caches = L_model_forward(X,parameters)

        cost = compute_cost(AL,Y)

        grads = L_model_backward(AL,Y,caches)

        parameters = update_parameters(parameters,grads,learning_rate)

        #打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
    #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters
```

我们现在开始加载数据集。

```python
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
```

数据集加载完成，开始正式训练：

```python
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True,isPlot=True)
```

训练结果：

```python
第 0 次迭代，成本值为： 0.715731513414
第 100 次迭代，成本值为： 0.674737759347
第 200 次迭代，成本值为： 0.660336543362
第 300 次迭代，成本值为： 0.646288780215
第 400 次迭代，成本值为： 0.629813121693
第 500 次迭代，成本值为： 0.606005622927
第 600 次迭代，成本值为： 0.569004126398
第 700 次迭代，成本值为： 0.519796535044
第 800 次迭代，成本值为： 0.464157167863
第 900 次迭代，成本值为： 0.408420300483
第 1000 次迭代，成本值为： 0.373154992161
第 1100 次迭代，成本值为： 0.30572374573
第 1200 次迭代，成本值为： 0.268101528477
第 1300 次迭代，成本值为： 0.238724748277
第 1400 次迭代，成本值为： 0.206322632579
第 1500 次迭代，成本值为： 0.179438869275
第 1600 次迭代，成本值为： 0.157987358188
第 1700 次迭代，成本值为： 0.142404130123
第 1800 次迭代，成本值为： 0.128651659979
第 1900 次迭代，成本值为： 0.112443149982
第 2000 次迭代，成本值为： 0.0850563103497
第 2100 次迭代，成本值为： 0.0575839119861
第 2200 次迭代，成本值为： 0.044567534547
第 2300 次迭代，成本值为： 0.038082751666
第 2400 次迭代，成本值为： 0.0344107490184
```

![L_layer_model train result](https://img-blog.csdn.net/20180331164304844?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

训练完成，我们看一下预测：

```python
pred_train = predict(train_x, train_y, parameters) #训练集
pred_test = predict(test_x, test_y, parameters) #测试集
```

预测结果：

```python
准确度为: 0.9952153110047847
准确度为: 0.78
```

就准确度而言，从70%到72%再到78%，可以看到的是准确度在一点点增加。

### 分析

我们可以看一看有哪些东西在L层模型中被错误地标记了，导致准确率没有提高。

```python
def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


print_mislabeled_images(classes, test_x, test_y, pred_test)
```

运行结果：

![mislabeled_indices ](https://img-blog.csdn.net/20180331170501319?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

分析一下我们就可以得知原因了： 
模型往往表现欠佳的几种类型的图像包括：

- 猫身体在一个不同的位置
- 猫出现在相似颜色的背景下
- 不同的猫的颜色和品种
- 相机角度
- 图片的亮度
- 比例变化（猫的图像非常大或很小）

### 相关库代码

#### lr_utils.py

```python
# lr_utils.py
import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```

#### dnn_utils.py

```python
# dnn_utils.py
import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
```

#### testCase.py

```python
#testCase.py
import numpy as np

def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    return A, W, b

def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])

    return Y, aL

def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_activation_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache

def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ( (A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches

def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads
```

## 【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第一周作业(1&2&3) - 初始化、正则化、梯度校验

### 开始之前

在这篇文章中，我们要干三件事：

```python
1. 初始化参数：
    1.1：使用0来初始化参数。
    1.2：使用随机数来初始化参数。
    1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
2. 正则化模型：
    2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
    2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。
```

我们说完了我们要干什么，我们这就开始吧。

我们就开始导入相关的库：

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化
import reg_utils    #第二部分，正则化
import gc_utils     #第三部分，梯度校验
#%matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

### 初始化参数

我们在初始化之前，我们来看看我们的数据集是怎样的：

#### 读取并绘制数据

```python
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
```

执行结果：

![dataset](https://img-blog.csdn.net/20180408105722387?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们将要建立一个分类器把蓝点和红点分开，在之前我们已经实现过一个3层的神经网络，我们将对它进行初始化：

我们将会尝试下面三种初始化方法:

- 初始化为0：在输入参数中全部初始化为0，参数名为initialization = “zeros”，核心代码： 
    `parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))`

- 初始化为随机数：把输入参数设置为随机值，权重初始化为大的随机值。参数名为initialization = “random”，核心代码： 
    `parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10`

- 抑梯度异常初始化：参见梯度消失和梯度爆炸的那一个视频，参数名为initialization = “he”，核心代码： 

  ​		`parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])`

首先我们来看看我们的模型是怎样的：

```python
def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_polt=True):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
        is_polt - 是否绘制梯度下降的曲线图
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]

    #选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else : 
        print("错误的初始化参数！程序退出")
        exit

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        a3 , cache = init_utils.forward_propagation(X,parameters)

        #计算成本        
        cost = init_utils.compute_loss(a3,Y)

        #反向传播
        grads = init_utils.backward_propagation(X,Y,cache)

        #更新参数
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        #记录成本
        if i % 1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))


    #学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    #返回学习完毕后的参数
    return parameters
```

模型我们可以简单地看一下，我们这就开始尝试一下这三种初始化。

#### 初始化为零

```python

def initialize_parameters_zeros(layers_dims):
    """
    将模型的参数全部设置为0

    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            bL - 偏置向量，维度为（layers_dims[L],1）
    """
    parameters = {}

    L = len(layers_dims) #网络层数

    for l in range(1,L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l],1))

        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))

    return parameters
```

我们这就来测试一下：

```python
parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
W1 = [[ 0.  0.  0.]
 [ 0.  0.  0.]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[ 0.  0.]]
b2 = [[ 0.]]
```

我们可以看到W和b全部被初始化为0了，那么我们使用这些参数来训练模型，结果会怎样呢？

```python
parameters = model(train_X, train_Y, initialization = "zeros",is_polt=True)
```

执行结果：

```python
第0次迭代，成本值为：0.69314718056
第1000次迭代，成本值为：0.69314718056
第2000次迭代，成本值为：0.69314718056
第3000次迭代，成本值为：0.69314718056
第4000次迭代，成本值为：0.69314718056
第5000次迭代，成本值为：0.69314718056
第6000次迭代，成本值为：0.69314718056
第7000次迭代，成本值为：0.69314718056
第8000次迭代，成本值为：0.69314718056
第9000次迭代，成本值为：0.69314718056
第10000次迭代，成本值为：0.69314718056
第11000次迭代，成本值为：0.69314718056
第12000次迭代，成本值为：0.69314718056
第13000次迭代，成本值为：0.69314718056
第14000次迭代，成本值为：0.69314718056
```

![init wit zeros](https://img-blog.csdn.net/20180408111016191?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

从上图中我们可以看到学习率一直没有变化，也就是说这个模型根本没有学习。我们来看看预测的结果怎么样：

```python
print ("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)
```

执行结果：

```python
训练集:
Accuracy: 0.5
测试集:
Accuracy: 0.5
```

性能确实很差，而且成本并没有真正降低，算法的性能也比随机猜测要好。为什么？让我们看看预测和决策边界的细节：

```python
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

执行结果：

```python
predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0]]
predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```

![Model with Zeros initialization](https://img-blog.csdn.net/2018040811162911?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

分类失败，该模型预测每个都为0。通常来说，零初始化都会导致神经网络无法打破对称性，最终导致的结果就是无论网络有多少层，最终只能得到和Logistic函数相同的效果。

#### 随机初始化

为了打破对称性，我们可以随机地把参数赋值。在随机初始化之后，每个神经元可以开始学习其输入的不同功能，我们还会设置比较大的参数值，看看会发生什么。

```python
def initialize_parameters_random(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """

    np.random.seed(3)               # 指定随机种子
    parameters = {}
    L = len(layers_dims)            # 层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10 #使用10倍缩放
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))

    return parameters
```

我们可以来测试一下：

```python
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
W1 = [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[-0.82741481 -6.27000677]]
b2 = [[ 0.]]
```

看起来这些参数都是比较大的，我们来看看实际运行会怎么样：

```python
parameters = model(train_X, train_Y, initialization = "random",is_polt=True)
print("训练集：")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)
```

执行结果：

```python
第0次迭代，成本值为：inf
第1000次迭代，成本值为：0.625098279396
第2000次迭代，成本值为：0.59812165967
第3000次迭代，成本值为：0.56384175723
第4000次迭代，成本值为：0.55017030492
第5000次迭代，成本值为：0.544463290966
第6000次迭代，成本值为：0.5374513807
第7000次迭代，成本值为：0.476404207407
第8000次迭代，成本值为：0.397814922951
第9000次迭代，成本值为：0.393476402877
第10000次迭代，成本值为：0.392029546188
第11000次迭代，成本值为：0.389245981351
第12000次迭代，成本值为：0.386154748571
第13000次迭代，成本值为：0.38498472891
第14000次迭代，成本值为：0.382782830835
训练集：
Accuracy: 0.83
测试集：
Accuracy: 0.86
[[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 1
  1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0 0 0
  0 0 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1
  1 0 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 0 0 1 0
  1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1
  0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 0 1 1
  0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0 1
  1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1
  1 1 1 0]]
[[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0
  1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1
  1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]
```

![init with random](https://img-blog.csdn.net/20180408141943795?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们来把图绘制出来，看看分类的结果是怎样的。

```python
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

执行结果： 

![Model with large random initialization](https://img-blog.csdn.net/20180408142417132?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们可以看到误差开始很高。这是因为由于具有较大的随机权重，最后一个激活(sigmoid)输出的结果非常接近于0或1，而当它出现错误时，它会导致非常高的损失。初始化参数如果没有很好地话会导致梯度消失、爆炸，这也会减慢优化算法。如果我们对这个网络进行更长时间的训练，我们将看到更好的结果，但是使用过大的随机数初始化会减慢优化的速度。

  总而言之，将权重初始化为非常大的时候其实效果并不好，下面我们试试小一点的参数值。

#### 抑梯度异常初始化
```python
def initialize_parameters_he(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """

    np.random.seed(3)               # 指定随机种子
    parameters = {}
    L = len(layers_dims)            # 层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))

    return parameters
```

我们来测试一下这个函数：

```python
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
W1 = [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
b2 = [[ 0.]]
```

这样我们就基本把参数W初始化到了1附近，我们来实际运行一下看看：

```python
parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
print("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集:")
init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)
```

执行结果：

```python
第0次迭代，成本值为：0.883053746342
第1000次迭代，成本值为：0.687982591973
第2000次迭代，成本值为：0.675128626452
第3000次迭代，成本值为：0.652611776889
第4000次迭代，成本值为：0.608295897057
第5000次迭代，成本值为：0.530494449172
第6000次迭代，成本值为：0.413864581707
第7000次迭代，成本值为：0.311780346484
第8000次迭代，成本值为：0.236962153303
第9000次迭代，成本值为：0.185972872092
第10000次迭代，成本值为：0.150155562804
第11000次迭代，成本值为：0.123250792923
第12000次迭代，成本值为：0.0991774654653
第13000次迭代，成本值为：0.0845705595402
第14000次迭代，成本值为：0.0735789596268
训练集:
Accuracy: 0.993333333333
测试集:
Accuracy: 0.96
```

![init with he](https://img-blog.csdn.net/20180408144005440?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们可以看到误差越来越小，我们来绘制一下预测的情况：

```python
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

![Model with He initialization](https://img-blog.csdn.net/20180408144104702?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

初始化的模型将蓝色和红色的点在少量的迭代中很好地分离出来，总结一下：

1. 不同的初始化方法可能导致性能最终不同
2. 随机初始化有助于打破对称，使得不同隐藏层的单元可以学习到不同的参数。
3. 初始化时，初始值不宜过大。
4. He初始化搭配ReLU激活函数常常可以得到不错的效果。

在深度学习中，如果数据集没有足够大的话，可能会导致一些过拟合的问题。过拟合导致的结果就是在训练集上有着很高的精确度，但是在遇到新的样本时，精确度下降会很严重。为了避免过拟合的问题，接下来我们要讲解的方式就是正则化。

### 正则化模型

问题描述：假设你现在是一个AI专家，你需要设计一个模型，可以用于推荐在足球场中守门员将球发至哪个位置可以让本队的球员抢到球的可能性更大。说白了，实际上就是一个二分类，一半是己方抢到球，一半就是对方抢到球，我们来看一下这个图：![football position](https://img-blog.csdn.net/20180408145205837?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 读取并绘制数据集

我们来加载并查看一下我们的数据集：

```python
train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)
```

执行结果：

![football dataset visiable](https://img-blog.csdn.net/20180408145415936?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 每一个点代表球落下的可能的位置，蓝色代表己方的球员会抢到球，红色代表对手的球员会抢到球，我们要做的就是使用模型来画出一条线，来找到适合我方球员能抢到球的位置。 
我们要做以下三件事，来对比出不同的模型的优劣：

1. 不使用正则化
2. 使用正则化 
   2.1 使用L2正则化 
   2.2 使用随机节点删除

我们来看一下我们的模型：

- 正则化模式 - 将lambd输入设置为非零值。 我们使用“lambd”而不是“lambda”，因为“lambda”是Python中的保留关键字。
- 随机删除节点 - 将keep_prob设置为小于1的值

```python
def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
        is_polt - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数
        keep_prob - 随机删除节点的概率
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]

    #初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        ##是否随机删除节点
        if keep_prob == 1:
            ###不随机删除节点
            a3 , cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            ###随机删除节点
            a3 , cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit

        #计算成本
        ## 是否使用二范数
        if lambd == 0:
            ###不使用L2正则化
            cost = reg_utils.compute_cost(a3,Y)
        else:
            ###使用L2正则化
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        #反向传播
        ##可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
        assert(lambd == 0  or keep_prob ==1)

        ##两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            ### 不使用L2正则化和不使用随机删除节点
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0:
            ### 使用L2正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            ### 使用随机删除节点，不使用L2正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        #更新参数
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

        #记录并打印成本
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                #打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    #是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    #返回学习后的参数
    return parameters
```

我们来先看一下不使用正则化下模型的效果：

#### 不使用正则化

```python
parameters = model(train_X, train_Y,is_plot=True)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
```

执行结果：

```python
第0次迭代，成本值为：0.655741252348
第10000次迭代，成本值为：0.163299875257
第20000次迭代，成本值为：0.138516424233
训练集:
Accuracy: 0.947867298578
测试集:
Accuracy: 0.915
```

![without reg](https://img-blog.csdn.net/20180408151147590?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们可以看到，对于训练集，精确度为94%；而对于测试集，精确度为91.5%。接下来，我们将分割曲线画出来：

```python
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

执行结果： ![Model without regularization](https://img-blog.csdn.net/20180408151058862?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

从图中可以看出，在无正则化时，分割曲线有了明显的过拟合特性。接下来，我们使用L2正则化：

#### 使用正则化

##### L2正则化

避免过度拟合的标准方法称为L2正则化，它包括适当修改你的成本函数。

我们下面就开始写相关的函数：

```python
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
    实现公式2的L2正则化计算成本

    参数：
        A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
        parameters - 包含模型学习后的参数的字典
    返回：
        cost - 使用公式2计算出来的正则化损失的值

    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3,Y)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))  + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

#当然，因为改变了成本函数，我们也必须改变向后传播的函数， 所有的梯度都必须根据这个新的成本值来计算。

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现我们添加了L2正则化的模型的后向传播。

    参数：
        X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
        Y - 标签，维度为（输出节点数量，数据集里面的数量）
        cache - 来自forward_propagation（）的cache输出
        lambda - regularization超参数，实数

    返回：
        gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
    """

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * np.dot(dZ3,A2.T) + ((lambd * W3) / m )
    db3 = (1 / m) * np.sum(dZ3,axis=1,keepdims=True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
```

我们来直接放到模型中跑一下：

```python
parameters = model(train_X, train_Y, lambd=0.7,is_plot=True)
print("使用正则化，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用正则化，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
```

执行结果：

```python
第0次迭代，成本值为：0.697448449313
第10000次迭代，成本值为：0.268491887328
第20000次迭代，成本值为：0.268091633713
使用正则化，训练集:
Accuracy: 0.938388625592
使用正则化，测试集:
Accuracy: 0.93
```

我们来看一下分类的结果

```python
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

执行结果： ![Model with L2-regularization](https://img-blog.csdn.net/20180408154106966?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

λ的值是可以使用开发集调整时的超参数。L2正则化会使决策边界更加平滑。如果λ太大，也可能会“过度平滑”，从而导致模型高偏差。L2正则化实际上在做什么？L2正则化依赖于较小权重的模型比具有较大权重的模型更简单这样的假设，因此，通过削减成本函数中权重的平方值，可以将所有权重值逐渐改变到到较小的值。权值数值高的话会有更平滑的模型，其中输入变化时输出变化更慢，但是你需要花费更多的时间。L2正则化对以下内容有影响：

- 成本计算       ： 正则化的计算需要添加到成本函数中
- 反向传播功能     ：在权重矩阵方面，梯度计算时也要依据正则化来做出相应的计算
- 重量变小（“重量衰减”) ：权重被逐渐改变到较小的值。

#### 随机删除节点

最后，我们使用Dropout来进行正则化，Dropout的原理就是每次迭代过程中随机将其中的一些节点失效。当我们关闭一些节点时，我们实际上修改了我们的模型。背后的想法是，在每次迭代时，我们都会训练一个只使用一部分神经元的不同模型。随着迭代次数的增加，我们的模型的节点会对其他特定节点的激活变得不那么敏感，因为其他节点可能在任何时候会失效。

```python
def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    """
    实现具有随机舍弃节点的前向传播。
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    参数：
        X  - 输入数据集，维度为（2，示例数）
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 权重矩阵，维度为（20,2）
            b1  - 偏向量，维度为（20,1）
            W2  - 权重矩阵，维度为（3,20）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）
        keep_prob  - 随机删除的概率，实数
    返回：
        A3  - 最后的激活值，维度为（1,1），正向传播的输出
        cache - 存储了一些用于计算反向传播的数值的元组
    """
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    #LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X) + b1
    A1 = reg_utils.relu(Z1)

    #下面的步骤1-4对应于上述的步骤1-4。
    D1 = np.random.rand(A1.shape[0],A1.shape[1])    #步骤1：初始化矩阵D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                             #步骤2：将D1的值转换为0或1（使​​用keep_prob作为阈值）
    A1 = A1 * D1                                    #步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A1 = A1 / keep_prob                             #步骤4：缩放未舍弃的节点(不为0)的值
    """
    #不理解的同学运行一下下面代码就知道了。
    import numpy as np
    np.random.seed(1)
    A1 = np.random.randn(1,3)

    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    keep_prob=0.5
    D1 = D1 < keep_prob
    print(D1)

    A1 = 0.01
    A1 = A1 * D1
    A1 = A1 / keep_prob
    print(A1)
    """

    Z2 = np.dot(W2,A1) + b2
    A2 = reg_utils.relu(Z2)

    #下面的步骤1-4对应于上述的步骤1-4。
    D2 = np.random.rand(A2.shape[0],A2.shape[1])    #步骤1：初始化矩阵D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                             #步骤2：将D2的值转换为0或1（使​​用keep_prob作为阈值）
    A2 = A2 * D2                                    #步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A2 = A2 / keep_prob                             #步骤4：缩放未舍弃的节点(不为0)的值

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache
```

改变了前向传播的算法，我们也需要改变后向传播的算法，使用存储在缓存中的掩码D[1] 和 D[2]将舍弃的节点位置信息添加到第一个和第二个隐藏层。

```python
def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
    实现我们随机删除的模型的后向传播。
    参数：
        X  - 输入数据集，维度为（2，示例数）
        Y  - 标签，维度为（输出节点数量，示例数量）
        cache - 来自forward_propagation_with_dropout（）的cache输出
        keep_prob  - 随机删除的概率，实数

    返回：
        gradients - 一个关于每个参数、激活值和预激活变量的梯度值的字典
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3,A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = dA2 * D2          # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1          # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
```

我们前向和后向传播的函数都写好了，现在用dropout运行模型（keep_prob = 0.86）跑一波。这意味着在每次迭代中，程序都可以24％的概率关闭第1层和第2层的每个神经元。调用的时候：

- 使用forward_propagation_with_dropout而不是forward_propagation。
- 使用backward_propagation_with_dropout而不是backward_propagation。

```python
parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)
```

执行结果：

```python
第0次迭代，成本值为：0.654391240515
第10000次迭代，成本值为：0.0610169865749
第20000次迭代，成本值为：0.0605824357985

使用随机删除节点，训练集:
Accuracy: 0.928909952607
使用随机删除节点，测试集:
Accuracy: 0.95
```

![train with dropout](https://img-blog.csdn.net/20180408170859845?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们来看看它的分类情况：

```python
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

执行结果：

![Model with dropout](https://img-blog.csdn.net/20180408171026699?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们可以看到，正则化会把训练集的准确度降低，但是测试集的准确度提高了，所以，我们这个还是成功了。

### 梯度校验

我们先来看一下一维线性模型的梯度检查计算过程： ![1Dgrad_kiank](https://img-blog.csdn.net/20180408172444832?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 一维线性

```python
def forward_propagation(x,theta):
    """

    实现图中呈现的线性前向传播（计算J）（J（theta）= theta * x）

    参数：
    x  - 一个实值输入
    theta  - 参数，也是一个实数

    返回：
    J  - 函数J的值，用公式J（theta）= theta * x计算
    """
    J = np.dot(theta,x)

    return J
```

测试一下：

```python
#测试forward_propagation
print("-----------------测试forward_propagation-----------------")
x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))
```

测试结果：

```python
-----------------测试forward_propagation-----------------
J = 8
```

前向传播有了，我们来看一下反向传播：

```python
def backward_propagation(x,theta):
    """
    计算J相对于θ的导数。

    参数：
        x  - 一个实值输入
        theta  - 参数，也是一个实数

    返回：
        dtheta  - 相对于θ的成本梯度
    """
    dtheta = x

    return dtheta
```

测试一下：

```python
#测试backward_propagation
print("-----------------测试backward_propagation-----------------")
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))
```

测试结果：

```python
-----------------测试backward_propagation-----------------
dtheta = 2
```

梯度检查的步骤如下：

1. $θ+=θ+ε$
2. $θ−=θ−ε$
3. $J+=J(θ+)$
4. $J−=J(θ−)$
5. $gradapprox=J+−J−2ε$

接下来，计算梯度的反向传播值，最后计算误差： 当difference小于10−7时，我们通常认为我们计算的结果是正确的。

```python
def gradient_check(x,theta,epsilon=1e-7):
    """

    实现图中的反向传播。

    参数：
        x  - 一个实值输入
        theta  - 参数，也是一个实数
        epsilon  - 使用公式（3）计算输入的微小偏移以计算近似梯度

    返回：
        近似梯度和后向传播梯度之间的差异
    """

    #使用公式（3）的左侧计算gradapprox。
    thetaplus = theta + epsilon                               # Step 1
    thetaminus = theta - epsilon                              # Step 2
    J_plus = forward_propagation(x, thetaplus)                # Step 3
    J_minus = forward_propagation(x, thetaminus)              # Step 4
    gradapprox = (J_plus - J_minus) / (2 * epsilon)           # Step 5


    #检查gradapprox是否足够接近backward_propagation（）的输出
    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)                      # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    # Step 2'
    difference = numerator / denominator                               # Step 3'

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference
```

测试一下：

```python
#测试gradient_check
print("-----------------测试gradient_check-----------------")
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))
```

测试结果：

```python
-----------------测试gradient_check-----------------
梯度检查：梯度正常!
difference = 2.91933588329e-10
```

高维参数是怎样计算的呢？我们看一下下图：

![NDgrad_kiank](https://img-blog.csdn.net/20180408173258633?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 高维

```python
def forward_propagation_n(X,Y,parameters):
    """
    实现图中的前向传播（并计算成本）。

    参数：
        X - 训练集为m个例子
        Y -  m个示例的标签
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 权重矩阵，维度为（5,4）
            b1  - 偏向量，维度为（5,1）
            W2  - 权重矩阵，维度为（3,5）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）

    返回：
        cost - 成本函数（logistic）
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X) + b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2,A1) + b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3,A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    #计算成本
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def backward_propagation_n(X,Y,cache):
    """
    实现图中所示的反向传播。

    参数：
        X - 输入数据点（输入节点数量，1）
        Y - 标签
        cache - 来自forward_propagation_n（）的cache输出

    返回：
        gradients - 一个字典，其中包含与每个参数、激活和激活前变量相关的成本梯度。
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3,A2.T)
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    #db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True) # Should not multiply by 4
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
```

```python
def gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7):
    """
    检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度

    参数：
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
        grad_output_propagation_n的输出包含与参数相关的成本梯度。
        x  - 输入数据点，维度为（输入节点数量，1）
        y  - 标签
        epsilon  - 计算输入的微小偏移以计算近似梯度

    返回：
        difference - 近似梯度和后向传播梯度之间的差异
    """
    #初始化参数
    parameters_values , keys = gc_utils.dictionary_to_vector(parameters) #keys用不到
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))

    #计算gradapprox
    for i in range(num_parameters):
        #计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
        thetaplus = np.copy(parameters_values)                                                  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                             # Step 2
        J_plus[i], cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaplus))  # Step 3 ，cache用不到

        #计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)                                                 # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                           # Step 2        
        J_minus[i], cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaminus))# Step 3 ，cache用不到

        #计算gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    #通过计算差异比较gradapprox和后向传播梯度。
    numerator = np.linalg.norm(grad - gradapprox)                                     # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # Step 2'
    difference = numerator / denominator                                              # Step 3'

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference
```

### 相关库代码

#### init_utils.py

```python
# -*- coding: utf-8 -*-

#init_utils.py

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)

    return s

def compute_loss(a3, Y):

    """
    Implement the loss function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    loss - value of the loss function
    """

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1./m * np.nansum(logprobs)

    return loss

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    """

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of n_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters['W' + str(i)] = ... 
                  parameters['b' + str(i)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]

    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    return p

def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions
```

#### reg_utils.py

```python
# -*- coding: utf-8 -*-

#reg_utils.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)

    return s


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])

    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)


    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    """

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache



def compute_cost(a3, Y):
    """
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    return cost

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of n_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters['W' + str(i)] = ... 
                  parameters['b' + str(i)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]

    return parameters




def load_2D_dataset(is_plot=True):
    data = sio.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    if is_plot:
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    return train_X, train_Y, test_X, test_Y

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    return p

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions
```

#### gc_utils.py

```python
# -*- coding: utf-8 -*-

#gc_utils.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)

    return s



def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta
```

## 【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第二周作业 - 优化算法

### 开始之前

  在正式开始之前，我们说一下我们要做什么。我们需要做的是分割数据集和优化梯度下降算法，所以我们需要做以下几件事： 
  1. 分割数据集 
  2. 优化梯度下降算法： 
     2.1 不使用任何优化算法 
     2.2 mini-batch梯度下降法 
     2.3 使用具有动量的梯度下降算法 
     2.4 使用Adam算法

   到目前为止，我们始终都是在使用梯度下降法学习，本文中，我们将使用一些更加高级的优化算法，利用这些优化算法，通常可以提高我们算法的收敛速度，并在最终得到更好的分离结果。这些方法可以加快学习速度，甚至可以为成本函数提供更好的最终值，在相同的结果下，有一个好的优化算法可以是等待几天和几个小时之间的差异。 
   我们想象一下成本函数J，最小化成本就像找到丘陵的最低点，在训练的每一步中，都会按照某个方向更新参数，以尽可能达到最低点。它类似于最快的下山的路，如下图：

![cost](https://img-blog.csdn.net/20180412093927807?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 导入库函数

```python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils #参见数据包或者在本文底部copy
import testCase  #参见数据包或者在本文底部copy

#%matplotlib inline #如果你用的是Jupyter Notebook请取消注释
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

### 梯度下降

在机器学习中，最简单就是没有任何优化的梯度下降(GD,Gradient Descent)，我们每一次循环都是对整个训练集进行学习，这叫做批量梯度下降(Batch Gradient Descent)，我们之前说过了最核心的参数更新的公式，这里我们再来看一下： 

$W[l]=W[l]−α dW[l]$

$b[l]=b[l]−α db[l]$

- l是指当前的层数
- α是学习率

所有的参数都在一个叫做parameters的字典类型的变量里面，我们来看看它怎样实现的吧，。

```python
def update_parameters_with_gd(parameters,grads,learning_rate):
    """
    使用梯度下降更新参数

    参数：
        parameters - 字典，包含了要更新的参数：
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads - 字典，包含了每一个梯度值用以更新参数
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate - 学习率

    返回值：
        parameters - 字典，包含了更新后的参数
    """

    L = len(parameters) // 2 #神经网络的层数

    #更新每个参数
    for l in range(L):
        parameters["W" + str(l +1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l +1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
```

我们来测试一下:

```python
#测试update_parameters_with_gd
print("-------------测试update_parameters_with_gd-------------")
parameters , grads , learning_rate = testCase.update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters,grads,learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
W1 = [[ 1.63535156 -0.62320365 -0.53718766]
 [-1.07799357  0.85639907 -2.29470142]]
b1 = [[ 1.74604067]
 [-0.75184921]]
W2 = [[ 0.32171798 -0.25467393  1.46902454]
 [-2.05617317 -0.31554548 -0.3756023 ]
 [ 1.1404819  -1.09976462 -0.1612551 ]]
b2 = [[-0.88020257]
 [ 0.02561572]
 [ 0.57539477]]
```

由梯度下降算法演变来的还有随机梯度下降（SGD）算法和小批量梯度下降算法，随机梯度下降（SGD），相当于小批量梯度下降，但是和mini-batch不同的是其中每个小批量(mini-batch)仅有1个样本，和梯度下降不同的是你一次只能在一个训练样本上计算梯度，而不是在整个训练集上计算梯度。我们来看一下它们的差异：

```python
#仅做比较，不运行。

#批量梯度下降，又叫梯度下降
X = data_input
Y = labels

parameters = initialize_parameters(layers_dims)
for i in range(0,num_iterations):
    #前向传播
    A,cache = forward_propagation(X,parameters)
    #计算损失
    cost = compute_cost(A,Y)
    #反向传播
    grads = backward_propagation(X,Y,cache)
    #更新参数
    parameters = update_parameters(parameters,grads)

#随机梯度下降算法：
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in (0,num_iterations):
    for j in m:
        #前向传播
        A,cache = forward_propagation(X,parameters)
        #计算成本
        cost = compute_cost(A,Y)
        #后向传播
        grads = backward_propagation(X,Y,cache)
        #更新参数
        parameters = update_parameters(parameters,grads)
```

在随机梯度下降算法中，每次迭代中仅使用其中一个样本，当训练集很大时，使用随机梯度下降算法的运行速度会很快，但是会存在一定的波动。 

在随机梯度下降中，在更新梯度之前，只使用1个训练样本。 当训练集较大时，随机梯度下降可以更快，但是参数会向最小值摆动，而不是平稳地收敛，我们来看一下比较图: 

![![kiank_minibatch](https://img-blog.csdn.net/20180412100218305?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)](https://img-blog.csdn.net/20180412100629968?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在实际中，更好的方法是使用小批量(mini-batch)梯度下降法，小批量梯度下降法是一种综合了梯度下降法和随机梯度下降法的方法，在它的每次迭代中，既不是选择全部的数据来学习，也不是选择一个样本来学习，而是把所有的数据集分割为一小块一小块的来学习，它会随机选择一小块（mini-batch），块大小一般为2的n次方倍。一方面，充分利用的GPU的并行性，更一方面，不会让计算时间特别长，来看一下比较图：![kiank_minibatch](https://img-blog.csdn.net/20180412100218305?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们现在从训练集中分割出mini-batch:

### mini-batch梯度下降法

我们要使用mini-batch要经过两个步骤： 

1. 把训练集打乱，但是X和Y依旧是一一对应的，之后，X的第i列是与Y中的第i个标签对应的样本。乱序步骤确保将样本被随机分成不同的小批次。如下图，X和Y的每一列代表一个样本 ![kiank_shuffle](https://img-blog.csdn.net/20180412102421401?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

切分，我们把训练集打乱之后，我们就可以对它进行切分了。这里切分的大小是64，如下图：![kiank_partition](https://img-blog.csdn.net/20180412102559746?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们先来看看分割后如何获取第一第二个mini-batch

```python
 #第一个mini-batch
 first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size]
 #第二个mini-batch
 second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]
 
 ...
```

这可能会有点不好理解，但是没关系，我们下面会有一些代码来帮你理解，我们首先来获取mini-batch

```python
 def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
     """
     从（X，Y）中创建一个随机的mini-batch列表
 
     参数：
         X - 输入数据，维度为(输入节点数量，样本的数量)
         Y - 对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
         mini_batch_size - 每个mini-batch的样本数量
 
     返回：
         mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）
 
     """
 
     np.random.seed(seed) #指定随机种子
     m = X.shape[1]
     mini_batches = []
 
     #第一步：打乱顺序
     permutation = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
     shuffled_X = X[:,permutation]   #将每一列的数据按permutation的顺序来重新排列。
     shuffled_Y = Y[:,permutation].reshape((1,m))
 
     """
     #博主注：
     #如果你不好理解的话请看一下下面的伪代码，看看X和Y是如何根据permutation来打乱顺序的。
     x = np.array([[1,2,3,4,5,6,7,8,9],
                   [9,8,7,6,5,4,3,2,1]])
     y = np.array([[1,0,1,0,1,0,1,0,1]])
 
     random_mini_batches(x,y)
     permutation= [7, 2, 1, 4, 8, 6, 3, 0, 5]
     shuffled_X= [[8 3 2 5 9 7 4 1 6]
                  [2 7 8 5 1 3 6 9 4]]
     shuffled_Y= [[0 1 0 1 1 1 0 1 0]]
     """
 
     #第二步，分割
     num_complete_minibatches = math.floor(m / mini_batch_size) #把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
     for k in range(0,num_complete_minibatches):
         mini_batch_X = shuffled_X[:,k * mini_batch_size:(k+1)*mini_batch_size]
         mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k+1)*mini_batch_size]
         """
         #博主注：
         #如果你不好理解的话请单独执行下面的代码，它可以帮你理解一些。
         a = np.array([[1,2,3,4,5,6,7,8,9],
                       [9,8,7,6,5,4,3,2,1],
                       [1,2,3,4,5,6,7,8,9]])
         k=1
         mini_batch_size=3
         print(a[:,1*3:(1+1)*3]) #从第4列到第6列
         '''
         [[4 5 6]
          [6 5 4]
          [4 5 6]]
         '''
         k=2
         print(a[:,2*3:(2+1)*3]) #从第7列到第9列
         '''
         [[7 8 9]
          [3 2 1]
          [7 8 9]]
         '''
 
         #看一下每一列的数据你可能就会好理解一些
         """
         mini_batch = (mini_batch_X,mini_batch_Y)
         mini_batches.append(mini_batch)
 
     #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
     #如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
     if m % mini_batch_size != 0:
         #获取最后剩余的部分
         mini_batch_X = shuffled_X[:,mini_batch_size * num_complete_minibatches:]
         mini_batch_Y = shuffled_Y[:,mini_batch_size * num_complete_minibatches:]
 
         mini_batch = (mini_batch_X,mini_batch_Y)
         mini_batches.append(mini_batch)
 
     return mini_batches
```

我们来测试一下：

```python
 #测试random_mini_batches
 print("-------------测试random_mini_batches-------------")
 X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case()
 mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)
 
 print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
 print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
 print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
 print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
 print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape)
 print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape)
 -------------测试random_mini_batches-------------
 第1个mini_batch_X 的维度为： (12288, 64)
 第1个mini_batch_Y 的维度为： (1, 64)
 第2个mini_batch_X 的维度为： (12288, 64)
 第2个mini_batch_Y 的维度为： (1, 64)
 第3个mini_batch_X 的维度为： (12288, 20)
 第3个mini_batch_Y 的维度为： (1, 20)
```

测试结果：

```python
 -------------测试random_mini_batches-------------
 第1个mini_batch_X 的维度为： (12288, 64)
 第1个mini_batch_Y 的维度为： (1, 64)
 第2个mini_batch_X 的维度为： (12288, 64)
 第2个mini_batch_Y 的维度为： (1, 64)
 第3个mini_batch_X 的维度为： (12288, 20)
 第3个mini_batch_Y 的维度为： (1, 20)
```

### 包含动量的梯度下降

由于小批量梯度下降只看到了一个子集的参数更新，更新的方向有一定的差异，所以小批量梯度下降的路径将“振荡地”走向收敛，使用动量可以减少这些振荡，动量考虑了过去的梯度以平滑更新， 我们将把以前梯度的方向存储在变量v中，从形式上讲，这将是前面的梯度的指数加权平均值。我们也可以把V看作是滚下坡的速度，根据山坡的坡度建立动量。我们来看一下下面的图： ![opt_momentum](https://img-blog.csdn.net/20180412104630539?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

既然我们要影响梯度的方向，而梯度需要使用到dW和db，那么我们就要建立一个和dW和db相同结构的变量来影响他们，我们现在来进行初始化：

```python
 def initialize_velocity(parameters):
     """
     初始化速度，velocity是一个字典：
         - keys: "dW1", "db1", ..., "dWL", "dbL" 
         - values:与相应的梯度/参数维度相同的值为零的矩阵。
     参数：
         parameters - 一个字典，包含了以下参数：
             parameters["W" + str(l)] = Wl
             parameters["b" + str(l)] = bl
     返回:
         v - 一个字典变量，包含了以下参数：
             v["dW" + str(l)] = dWl的速度
             v["db" + str(l)] = dbl的速度
 
     """
     L = len(parameters) // 2 #神经网络的层数
     v = {}
 
     for l in range(L):
         v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
         v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
 
     return v
```

我们来测试一下：

```python
 #测试initialize_velocity
 print("-------------测试initialize_velocity-------------")
 parameters = testCase.initialize_velocity_test_case()
 v = initialize_velocity(parameters)
 
 print('v["dW1"] = ' + str(v["dW1"]))
 print('v["db1"] = ' + str(v["db1"]))
 print('v["dW2"] = ' + str(v["dW2"]))
 print('v["db2"] = ' + str(v["db2"]))
```

测试结果：

```python
 -------------测试initialize_velocity-------------
 v["dW1"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]]
 v["db1"] = [[ 0.]
  [ 0.]]
 v["dW2"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]
  [ 0.  0.  0.]]
 v["db2"] = [[ 0.]
  [ 0.]
  [ 0.]]
```

既然初始化完成了，我们就开始影响梯度的方向。

```python
 def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
     """
     使用动量更新参数
     参数：
         parameters - 一个字典类型的变量，包含了以下字段：
             parameters["W" + str(l)] = Wl
             parameters["b" + str(l)] = bl
         grads - 一个包含梯度值的字典变量，具有以下字段：
             grads["dW" + str(l)] = dWl
             grads["db" + str(l)] = dbl
         v - 包含当前速度的字典变量，具有以下字段：
             v["dW" + str(l)] = ...
             v["db" + str(l)] = ...
         beta - 超参数，动量，实数
         learning_rate - 学习率，实数
     返回：
         parameters - 更新后的参数字典
         v - 包含了更新后的速度变量
     """
     L = len(parameters) // 2 
     for l in range(L):
         #计算速度
         v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
         v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
 
         #更新参数
         parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
         parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
 
     return parameters,v
```

我们来测试一下：

```python
 #测试update_parameters_with_momentun
 print("-------------测试update_parameters_with_momentun-------------")
 parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
 update_parameters_with_momentun(parameters,grads,v,beta=0.9,learning_rate=0.01)
 
 print("W1 = " + str(parameters["W1"]))
 print("b1 = " + str(parameters["b1"]))
 print("W2 = " + str(parameters["W2"]))
 print("b2 = " + str(parameters["b2"]))
 print('v["dW1"] = ' + str(v["dW1"]))
 print('v["db1"] = ' + str(v["db1"]))
 print('v["dW2"] = ' + str(v["dW2"]))
 print('v["db2"] = ' + str(v["db2"]))
```

测试结果

```python
 -------------测试update_parameters_with_momentun-------------
 W1 = [[ 1.62544598 -0.61290114 -0.52907334]
  [-1.07347112  0.86450677 -2.30085497]]
 b1 = [[ 1.74493465]
  [-0.76027113]]
 W2 = [[ 0.31930698 -0.24990073  1.4627996 ]
  [-2.05974396 -0.32173003 -0.38320915]
  [ 1.13444069 -1.0998786  -0.1713109 ]]
 b2 = [[-0.87809283]
  [ 0.04055394]
  [ 0.58207317]]
 v["dW1"] = [[-0.11006192  0.11447237  0.09015907]
  [ 0.05024943  0.09008559 -0.06837279]]
 v["db1"] = [[-0.01228902]
  [-0.09357694]]
 v["dW2"] = [[-0.02678881  0.05303555 -0.06916608]
  [-0.03967535 -0.06871727 -0.08452056]
  [-0.06712461 -0.00126646 -0.11173103]]
 v["db2"] = [[ 0.02344157]
  [ 0.16598022]
  [ 0.07420442]]
```

需要注意的是速度v是用0来初始化的，因此，该算法需要经过几次迭代才能把速度提升上来并开始跨越更大步伐。当beta=0时，该算法相当于是没有使用momentum算法的标准的梯度下降算法。当beta越大的时候，说明平滑的作用越明显。通常0.9是比较合适的值。那如何才能在开始的时候就保持很快的速度向最小误差那里前进呢？我们来看看下面的Adam算法。

### Adam算法

Adam算法是训练神经网络中最有效的算法之一，它是RMSProp算法与Momentum算法的结合体。

1. 计算以前的梯度的指数加权平均值，并将其存储在变量v （偏差校正前）和vcorrected （偏差校正后）中。 
2. 计算以前梯度的平方的指数加权平均值，并将其存储在变量s （偏差校正前）和scorrected （偏差校正后）中。 
3. 根据1和2更新参数。 

我们先来初始化Adam所需要的参数：

```python
 def initialize_adam(parameters):
     """
     初始化v和s，它们都是字典类型的变量，都包含了以下字段：
         - keys: "dW1", "db1", ..., "dWL", "dbL" 
         - values：与对应的梯度/参数相同维度的值为零的numpy矩阵
 
     参数：
         parameters - 包含了以下参数的字典变量：
             parameters["W" + str(l)] = Wl
             parameters["b" + str(l)] = bl
     返回：
         v - 包含梯度的指数加权平均值，字段如下：
             v["dW" + str(l)] = ...
             v["db" + str(l)] = ...
         s - 包含平方梯度的指数加权平均值，字段如下：
             s["dW" + str(l)] = ...
             s["db" + str(l)] = ...
 
     """
 
     L = len(parameters) // 2
     v = {}
     s = {}
 
     for l in range(L):
         v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
         v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
 
         s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
         s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
 
     return (v,s)
```

测试一下：

```python
 #测试initialize_adam
 print("-------------测试initialize_adam-------------")
 parameters = testCase.initialize_adam_test_case()
 v,s = initialize_adam(parameters)
 
 print('v["dW1"] = ' + str(v["dW1"])) 
 print('v["db1"] = ' + str(v["db1"])) 
 print('v["dW2"] = ' + str(v["dW2"])) 
 print('v["db2"] = ' + str(v["db2"])) 
 print('s["dW1"] = ' + str(s["dW1"])) 
 print('s["db1"] = ' + str(s["db1"])) 
 print('s["dW2"] = ' + str(s["dW2"])) 
 print('s["db2"] = ' + str(s["db2"])) 
```

测试结果：

```python
 -------------测试initialize_adam-------------
 v["dW1"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]]
 v["db1"] = [[ 0.]
  [ 0.]]
 v["dW2"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]
  [ 0.  0.  0.]]
 v["db2"] = [[ 0.]
  [ 0.]
  [ 0.]]
 s["dW1"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]]
 s["db1"] = [[ 0.]
  [ 0.]]
 s["dW2"] = [[ 0.  0.  0.]
  [ 0.  0.  0.]
  [ 0.  0.  0.]]
 s["db2"] = [[ 0.]
  [ 0.]
  [ 0.]]
```

参数初始化完成了，我们就根据公式来更新参数： 

```python
 def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
     """
     使用Adam更新参数
 
     参数：
         parameters - 包含了以下字段的字典：
             parameters['W' + str(l)] = Wl
             parameters['b' + str(l)] = bl
         grads - 包含了梯度值的字典，有以下key值：
             grads['dW' + str(l)] = dWl
             grads['db' + str(l)] = dbl
         v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
         s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
         t - 当前迭代的次数
         learning_rate - 学习率
         beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
         beta2 - RMSprop的一个参数，超参数
         epsilon - 防止除零操作（分母为0）
 
     返回：
         parameters - 更新后的参数
         v - 第一个梯度的移动平均值，是一个字典类型的变量
         s - 平方梯度的移动平均值，是一个字典类型的变量
     """
     L = len(parameters) // 2
     v_corrected = {} #偏差修正后的值
     s_corrected = {} #偏差修正后的值
 
     for l in range(L):
         #梯度的移动平均值,输入："v , grads , beta1",输出：" v "
         v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
         v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
 
         #计算第一阶段的偏差修正后的估计值，输入"v , beta1 , t" , 输出："v_corrected"
         v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1,t))
         v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1,t))
 
         #计算平方梯度的移动平均值，输入："s, grads , beta2"，输出："s"
         s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
         s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
 
         #计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
         s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2,t))
         s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2,t))
 
         #更新参数，输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
         parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
         parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))
 
     return (parameters,v,s)
```

测试一下：

```python
 #测试update_with_parameters_with_adam
 print("-------------测试update_with_parameters_with_adam-------------")
 parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
 update_parameters_with_adam(parameters,grads,v,s,t=2)
 
 print("W1 = " + str(parameters["W1"]))
 print("b1 = " + str(parameters["b1"]))
 print("W2 = " + str(parameters["W2"]))
 print("b2 = " + str(parameters["b2"]))
 print('v["dW1"] = ' + str(v["dW1"])) 
 print('v["db1"] = ' + str(v["db1"])) 
 print('v["dW2"] = ' + str(v["dW2"])) 
 print('v["db2"] = ' + str(v["db2"])) 
 print('s["dW1"] = ' + str(s["dW1"])) 
 print('s["db1"] = ' + str(s["db1"])) 
 print('s["dW2"] = ' + str(s["dW2"])) 
 print('s["db2"] = ' + str(s["db2"])) 
```

测试结果：

```python
 -------------测试update_with_parameters_with_adam-------------
 W1 = [[ 1.63178673 -0.61919778 -0.53561312]
  [-1.08040999  0.85796626 -2.29409733]]
 b1 = [[ 1.75225313]
  [-0.75376553]]
 W2 = [[ 0.32648046 -0.25681174  1.46954931]
  [-2.05269934 -0.31497584 -0.37661299]
  [ 1.14121081 -1.09245036 -0.16498684]]
 b2 = [[-0.88529978]
  [ 0.03477238]
  [ 0.57537385]]
 v["dW1"] = [[-0.11006192  0.11447237  0.09015907]
  [ 0.05024943  0.09008559 -0.06837279]]
 v["db1"] = [[-0.01228902]
  [-0.09357694]]
 v["dW2"] = [[-0.02678881  0.05303555 -0.06916608]
  [-0.03967535 -0.06871727 -0.08452056]
  [-0.06712461 -0.00126646 -0.11173103]]
 v["db2"] = [[ 0.02344157]
  [ 0.16598022]
  [ 0.07420442]]
 s["dW1"] = [[ 0.00121136  0.00131039  0.00081287]
  [ 0.0002525   0.00081154  0.00046748]]
 s["db1"] = [[  1.51020075e-05]
  [  8.75664434e-04]]
 s["dW2"] = [[  7.17640232e-05   2.81276921e-04   4.78394595e-04]
  [  1.57413361e-04   4.72206320e-04   7.14372576e-04]
  [  4.50571368e-04   1.60392066e-07   1.24838242e-03]]
 s["db2"] = [[  5.49507194e-05]
  [  2.75494327e-03]
  [  5.50629536e-04]]
```

现在我们三个优化器都做好了，我们这就来看看效果到底怎么样：

### 测试

在测试正式开始之前，我们需要把数据集加载进来。

#### 加载数据集

我们使用下面的“月亮（moon）”数据集来测试不同的优化方法。数据集被命名为“月亮”，因为这两个类的数据看起来有点像新月形的月亮。

```python
 train_X, train_Y = opt_utils.load_dataset(is_plot=True)
```

![dataset_image_mall](https://img-blog.csdn.net/2018041211350759?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们之前已经实现过了一个三层的神经网络，我们将分别用它来测试我们的优化器的优化效果，我们先来看看我们的模型是什么样的：

#### 定义模型

```python
 def model(X,Y,layers_dims,optimizer,learning_rate=0.0007,
           mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,
           epsilon=1e-8,num_epochs=10000,print_cost=True,is_plot=True):
 
     """
     可以运行在不同优化器模式下的3层神经网络模型。
 
     参数：
         X - 输入数据，维度为（2，输入的数据集里面样本数量）
         Y - 与X对应的标签
         layers_dims - 包含层数和节点数量的列表
         optimizer - 字符串类型的参数，用于选择优化类型，【 "gd" | "momentum" | "adam" 】
         learning_rate - 学习率
         mini_batch_size - 每个小批量数据集的大小
         beta - 用于动量优化的一个超参数
         beta1 - 用于计算梯度后的指数衰减的估计的超参数
         beta1 - 用于计算平方梯度后的指数衰减的估计的超参数
         epsilon - 用于在Adam中避免除零操作的超参数，一般不更改
         num_epochs - 整个训练集的遍历次数，（视频2.9学习率衰减，1分55秒处，视频中称作“代”）,相当于之前的num_iteration
         print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
         is_plot - 是否绘制出曲线图
 
     返回：
         parameters - 包含了学习后的参数
 
     """
     L = len(layers_dims)
     costs = []
     t = 0 #每学习完一个minibatch就增加1
     seed = 10 #随机种子
 
     #初始化参数
     parameters = opt_utils.initialize_parameters(layers_dims)
 
     #选择优化器
     if optimizer == "gd":
         pass #不使用任何优化器，直接使用梯度下降法
     elif optimizer == "momentum":
         v = initialize_velocity(parameters) #使用动量
     elif optimizer == "adam":
         v, s = initialize_adam(parameters)#使用Adam优化
     else:
         print("optimizer参数错误，程序退出。")
         exit(1)
 
     #开始学习
     for i in range(num_epochs):
         #定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
         seed = seed + 1
         minibatches = random_mini_batches(X,Y,mini_batch_size,seed)
 
         for minibatch in minibatches:
             #选择一个minibatch
             (minibatch_X,minibatch_Y) = minibatch
 
             #前向传播
             A3 , cache = opt_utils.forward_propagation(minibatch_X,parameters)
 
             #计算误差
             cost = opt_utils.compute_cost(A3 , minibatch_Y)
 
             #反向传播
             grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)
 
             #更新参数
             if optimizer == "gd":
                 parameters = update_parameters_with_gd(parameters,grads,learning_rate)
             elif optimizer == "momentum":
                 parameters, v = update_parameters_with_momentun(parameters,grads,v,beta,learning_rate)
             elif optimizer == "adam":
                 t = t + 1 
                 parameters , v , s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
         #记录误差值
         if i % 100 == 0:
             costs.append(cost)
             #是否打印误差值
             if print_cost and i % 1000 == 0:
                 print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
     #是否绘制曲线图
     if is_plot:
         plt.plot(costs)
         plt.ylabel('cost')
         plt.xlabel('epochs (per 100)')
         plt.title("Learning rate = " + str(learning_rate))
         plt.show()
 
     return parameters
```

模型已经有了，我们就先来测试没有任何优化的梯度下降：

#### 梯度下降测试

```python
 #使用普通的梯度下降
 layers_dims = [train_X.shape[0],5,2,1]
 parameters = model(train_X, train_Y, layers_dims, optimizer="gd",is_plot=True)
```

运行结果： ![gd](https://img-blog.csdn.net/20180412113923319?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```python
 第0次遍历整个数据集，当前误差值：0.690735512291
 第1000次遍历整个数据集，当前误差值：0.685272532846
 第2000次遍历整个数据集，当前误差值：0.647072224072
 第3000次遍历整个数据集，当前误差值：0.619524554997
 第4000次遍历整个数据集，当前误差值：0.576584435595
 第5000次遍历整个数据集，当前误差值：0.607242639597
 第6000次遍历整个数据集，当前误差值：0.529403331768
 第7000次遍历整个数据集，当前误差值：0.460768239859
 第8000次遍历整个数据集，当前误差值：0.465586082399
 第9000次遍历整个数据集，当前误差值：0.464517972217
```

我们来绘制分类的情况：

```python
 #预测
 preditions = opt_utils.predict(train_X,train_Y,parameters)
 
 #绘制分类图
 plt.title("Model with Gradient Descent optimization")
 axes = plt.gca()
 axes.set_xlim([-1.5, 2.5])
 axes.set_ylim([-1, 1.5])
 opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

运行结果：

```python
 Accuracy: 0.796666666667
```

![Model with Gradient Descent optimization](https://img-blog.csdn.net/20180412114119717?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 具有动量的梯度下降测试

因为这个例子比较简单，使用动量效果很小，但对于更复杂的问题，你可能会看到更好的效果。

```python
 layers_dims = [train_X.shape[0],5,2,1]
 #使用动量的梯度下降
 parameters = model(train_X, train_Y, layers_dims, beta=0.9,optimizer="momentum",is_plot=True)
```

运行结果：

```python
 第0次遍历整个数据集，当前误差值：0.690741298835
 第1000次遍历整个数据集，当前误差值：0.685340526127
 第2000次遍历整个数据集，当前误差值：0.64714483701
 第3000次遍历整个数据集，当前误差值：0.619594303208
 第4000次遍历整个数据集，当前误差值：0.576665034407
 第5000次遍历整个数据集，当前误差值：0.607323821901
 第6000次遍历整个数据集，当前误差值：0.529476175879
 第7000次遍历整个数据集，当前误差值：0.460936190049
 第8000次遍历整个数据集，当前误差值：0.465780093701
 第9000次遍历整个数据集，当前误差值：0.464739596792
```

![momentum](https://img-blog.csdn.net/20180412114402891?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们来绘制分类的情况：

```python
 #预测
 preditions = opt_utils.predict(train_X,train_Y,parameters)
 
 #绘制分类图
 plt.title("Model with Momentum optimization")
 axes = plt.gca()
 axes.set_xlim([-1.5, 2.5])
 axes.set_ylim([-1, 1.5])
 opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

运行结果： 

![Model with Momentum optimization](https://img-blog.csdn.net/20180412114505722?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们再来看看Adam的运行效果：

#### Adam优化后的梯度下降

```python
 layers_dims = [train_X.shape[0], 5, 2, 1]
 #使用Adam优化的梯度下降
 parameters = model(train_X, train_Y, layers_dims, optimizer="adam",is_plot=True)
```

运行结果：

```python
 第0次遍历整个数据集，当前误差值：0.690552244611
 第1000次遍历整个数据集，当前误差值：0.185501364386
 第2000次遍历整个数据集，当前误差值：0.150830465753
 第3000次遍历整个数据集，当前误差值：0.07445438571
 第4000次遍历整个数据集，当前误差值：0.125959156513
 第5000次遍历整个数据集，当前误差值：0.104344435342
 第6000次遍历整个数据集，当前误差值：0.100676375041
 第7000次遍历整个数据集，当前误差值：0.0316520301351
 第8000次遍历整个数据集，当前误差值：0.111972731312
 第9000次遍历整个数据集，当前误差值：0.197940071525
```

![adam](https://img-blog.csdn.net/20180412114709537?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们再来看看分类的效果图：

```python
 #预测
 preditions = opt_utils.predict(train_X,train_Y,parameters)
 
 #绘制分类图
 plt.title("Model with Adam optimization")
 axes = plt.gca()
 axes.set_xlim([-1.5, 2.5])
 axes.set_ylim([-1, 1.5])
 opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
```

运行结果：

```python
 Accuracy: 0.94
```

![Model with Adam optimization](https://img-blog.csdn.net/20180412114915803?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 总结

| 优化算法               | 准确度 | 曲线平滑度 |
| :--------------------- | :----- | :--------- |
| 梯度下降               | 79.7%  | 震荡       |
| 具有动量的梯度下降算法 | 79.7%  | 震荡       |
| Adam优化后的梯度下降   | 94%    | 平滑       |

具有动量的梯度下降通常可以有很好的效果，但由于小的学习速率和简单的数据集所以它的影响几乎是轻微的。另一方面，Adam明显优于小批量梯度下降和具有动量的梯度下降，如果在这个简单的模型上运行更多时间的数据集，这三种方法都会产生非常好的结果，然而，我们已经看到Adam收敛得更快。

  Adam的一些优点包括相对较低的内存要求（虽然比梯度下降和动量下降更高）和通常运作良好，即使对参数进行微调（除了学习率α）

### 相关库代码

#### opt_utils.py

```python
 # -*- coding: utf-8 -*-
 
 #opt_utils.py
 
 import numpy as np
 import matplotlib.pyplot as plt
 import sklearn
 import sklearn.datasets
 
 def sigmoid(x):
     """
     Compute the sigmoid of x
 
     Arguments:
     x -- A scalar or numpy array of any size.
 
     Return:
     s -- sigmoid(x)
     """
     s = 1/(1+np.exp(-x))
     return s
 
 def relu(x):
     """
     Compute the relu of x
 
     Arguments:
     x -- A scalar or numpy array of any size.
 
     Return:
     s -- relu(x)
     """
     s = np.maximum(0,x)
 
     return s
 
 
 def load_params_and_grads(seed=1):
     np.random.seed(seed)
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
 
     dW1 = np.random.randn(2,3)
     db1 = np.random.randn(2,1)
     dW2 = np.random.randn(3,3)
     db2 = np.random.randn(3,1)
 
     return W1, b1, W2, b2, dW1, db1, dW2, db2
 
 def initialize_parameters(layer_dims):
     """
     Arguments:
     layer_dims -- python array (list) containing the dimensions of each layer in our network
 
     Returns:
     parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                     W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                     b1 -- bias vector of shape (layer_dims[l], 1)
                     Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                     bl -- bias vector of shape (1, layer_dims[l])
 
     Tips:
     - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
     This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
     - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
     """
 
     np.random.seed(3)
     parameters = {}
     L = len(layer_dims) # number of layers in the network
 
     for l in range(1, L):
         parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
         parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
 
         assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
         assert(parameters['W' + str(l)].shape == layer_dims[l], 1)
 
     return parameters
 
 def forward_propagation(X, parameters):
     """
     Implements the forward propagation (and computes the loss) presented in Figure 2.
 
     Arguments:
     X -- input dataset, of shape (input size, number of examples)
     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                     W1 -- weight matrix of shape ()
                     b1 -- bias vector of shape ()
                     W2 -- weight matrix of shape ()
                     b2 -- bias vector of shape ()
                     W3 -- weight matrix of shape ()
                     b3 -- bias vector of shape ()
 
     Returns:
     loss -- the loss function (vanilla logistic loss)
     """
 
     # retrieve parameters
     W1 = parameters["W1"]
     b1 = parameters["b1"]
     W2 = parameters["W2"]
     b2 = parameters["b2"]
     W3 = parameters["W3"]
     b3 = parameters["b3"]
 
     # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
     z1 = np.dot(W1, X) + b1
     a1 = relu(z1)
     z2 = np.dot(W2, a1) + b2
     a2 = relu(z2)
     z3 = np.dot(W3, a2) + b3
     a3 = sigmoid(z3)
 
     cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
 
     return a3, cache
 
 def backward_propagation(X, Y, cache):
     """
     Implement the backward propagation presented in figure 2.
 
     Arguments:
     X -- input dataset, of shape (input size, number of examples)
     Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
     cache -- cache output from forward_propagation()
 
     Returns:
     gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
     """
     m = X.shape[1]
     (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
 
     dz3 = 1./m * (a3 - Y)
     dW3 = np.dot(dz3, a2.T)
     db3 = np.sum(dz3, axis=1, keepdims = True)
 
     da2 = np.dot(W3.T, dz3)
     dz2 = np.multiply(da2, np.int64(a2 > 0))
     dW2 = np.dot(dz2, a1.T)
     db2 = np.sum(dz2, axis=1, keepdims = True)
 
     da1 = np.dot(W2.T, dz2)
     dz1 = np.multiply(da1, np.int64(a1 > 0))
     dW1 = np.dot(dz1, X.T)
     db1 = np.sum(dz1, axis=1, keepdims = True)
 
     gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                  "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                  "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
 
     return gradients
 
 def compute_cost(a3, Y):
 
     """
     Implement the cost function
 
     Arguments:
     a3 -- post-activation, output of forward propagation
     Y -- "true" labels vector, same shape as a3
 
     Returns:
     cost - value of the cost function
     """
     m = Y.shape[1]
 
     logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
     cost = 1./m * np.sum(logprobs)
 
     return cost
 
 def predict(X, y, parameters):
     """
     This function is used to predict the results of a  n-layer neural network.
 
     Arguments:
     X -- data set of examples you would like to label
     parameters -- parameters of the trained model
 
     Returns:
     p -- predictions for the given dataset X
     """
 
     m = X.shape[1]
     p = np.zeros((1,m), dtype = np.int)
 
     # Forward propagation
     a3, caches = forward_propagation(X, parameters)
 
     # convert probas to 0/1 predictions
     for i in range(0, a3.shape[1]):
         if a3[0,i] > 0.5:
             p[0,i] = 1
         else:
             p[0,i] = 0
 
     # print results
 
     #print ("predictions: " + str(p[0,:]))
     #print ("true labels: " + str(y[0,:]))
     print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
 
     return p
 
 def predict_dec(parameters, X):
     """
     Used for plotting decision boundary.
 
     Arguments:
     parameters -- python dictionary containing your parameters 
     X -- input data of size (m, K)
 
     Returns
     predictions -- vector of predictions of our model (red: 0 / blue: 1)
     """
 
     # Predict using forward propagation and a classification threshold of 0.5
     a3, cache = forward_propagation(X, parameters)
     predictions = (a3 > 0.5)
     return predictions
 
 def plot_decision_boundary(model, X, y):
     # Set min and max values and give it some padding
     x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
     y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
     h = 0.01
     # Generate a grid of points with distance h between them
     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
     # Predict the function value for the whole grid
     Z = model(np.c_[xx.ravel(), yy.ravel()])
     Z = Z.reshape(xx.shape)
     # Plot the contour and training examples
     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
     plt.ylabel('x2')
     plt.xlabel('x1')
     plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
     plt.show()
 
 def load_dataset(is_plot = True):
     np.random.seed(3)
     train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
     # Visualize the data
     if is_plot:
         plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
     train_X = train_X.T
     train_Y = train_Y.reshape((1, train_Y.shape[0]))
 
     return train_X, train_Y
```

#### testCase.py

```python
 # -*- coding: utf-8 -*-
 
 #testCase.py
 
 import numpy as np
 
 def update_parameters_with_gd_test_case():
     np.random.seed(1)
     learning_rate = 0.01
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
 
     dW1 = np.random.randn(2,3)
     db1 = np.random.randn(2,1)
     dW2 = np.random.randn(3,3)
     db2 = np.random.randn(3,1)
 
     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
 
     return parameters, grads, learning_rate
 
 """
 def update_parameters_with_sgd_checker(function, inputs, outputs):
     if function(inputs) == outputs:
         print("Correct")
     else:
         print("Incorrect")
 """
 
 def random_mini_batches_test_case():
     np.random.seed(1)
     mini_batch_size = 64
     X = np.random.randn(12288, 148)
     Y = np.random.randn(1, 148) < 0.5
     return X, Y, mini_batch_size
 
 def initialize_velocity_test_case():
     np.random.seed(1)
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
     return parameters
 
 def update_parameters_with_momentum_test_case():
     np.random.seed(1)
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
 
     dW1 = np.random.randn(2,3)
     db1 = np.random.randn(2,1)
     dW2 = np.random.randn(3,3)
     db2 = np.random.randn(3,1)
     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
     v = {'dW1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])}
     return parameters, grads, v
 
 def initialize_adam_test_case():
     np.random.seed(1)
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
     return parameters
 
 def update_parameters_with_adam_test_case():
     np.random.seed(1)
     v, s = ({'dW1': np.array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
          [ 0.]]), 'db2': np.array([[ 0.],
          [ 0.],
          [ 0.]])}, {'dW1': np.array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
          [ 0.]]), 'db2': np.array([[ 0.],
          [ 0.],
          [ 0.]])})
     W1 = np.random.randn(2,3)
     b1 = np.random.randn(2,1)
     W2 = np.random.randn(3,3)
     b2 = np.random.randn(3,1)
 
     dW1 = np.random.randn(2,3)
     db1 = np.random.randn(2,1)
     dW2 = np.random.randn(3,3)
     db2 = np.random.randn(3,1)
 
     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
 
     return parameters, grads, v, s
```

## 【吴恩达课后编程作业】Course 2 - 改善深层神经网络 - 第三周作业 - TensorFlow入门

### TensorFlow 入门

  到目前为止，我们一直在使用numpy来自己编写神经网络。现在我们将一步步的使用深度学习的框架来很容易的构建属于自己的神经网络。我们将学习TensorFlow这个框架：

- 初始化变量
- 建立一个会话
- 训练的算法
- 实现一个神经网络

使用框架编程不仅可以节省你的写代码时间，还可以让你的优化速度更快。

#### 1 - 导入TensorFlow库

开始之前，我们先导入一些库

```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

#%matplotlib inline #如果你使用的是jupyter notebook取消注释
np.random.seed(1)
```

我们现在已经导入了相关的库，我们将引导你完成不同的应用，我们现在看一下下面的计算损失的公式： 

$loss=L(y^,y)=(y^(i)−y(i))2$

```python
y_hat = tf.constant(36,name="y_hat")            #定义y_hat为固定值36
y = tf.constant(39,name="y")                    #定义y为固定值39

loss = tf.Variable((y-y_hat)**2,name="loss" )   #为损失函数创建一个变量

init = tf.global_variables_initializer()        #运行之后的初始化(ession.run(init))
                                                #损失变量将被初始化并准备计算
with tf.Session() as session:                   #创建一个session并打印输出
    session.run(init)                           #初始化变量
    print(session.run(loss))                    #打印损失值
```

执行结果：

```python
9
```

对于Tensorflow的代码实现而言，实现代码的结构如下：

1. 创建Tensorflow变量（此时，尚未直接计算）
2. 实现Tensorflow变量之间的操作定义
3. 初始化Tensorflow变量
4. 创建Session
5. 运行Session，此时，之前编写操作都会在这一步运行。

  因此，当我们为损失函数创建一个变量时，我们简单地将损失定义为其他数量的函数，但没有评估它的价值。 为了评估它，我们需要运行init=tf.global_variables_initializer()，初始化损失变量，在最后一行，我们最后能够评估损失的值并打印它的值。

现在让我们看一个简单的例子：

```python
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)

print(c)
```

执行结果：

```python
Tensor("Mul:0", shape=(), dtype=int32)
```

正如预料中一样，我们并没有看到结果20，不过我们得到了一个Tensor类型的变量，没有维度，数字类型为int32。我们之前所做的一切都只是把这些东西放到了一个“计算图(computation graph)”中，而我们还没有开始运行这个计算图，为了实际计算这两个数字，我们需要创建一个会话并运行它：

```python
sess = tf.Session()

print(sess.run(c))
```

执行结果：

```python
20
```

总结一下，记得初始化变量，然后创建一个session来运行它。 

接下来，我们需要了解一下占位符（placeholders）。占位符是一个对象，它的值只能在稍后指定，要指定占位符的值，可以使用一个feed字典（feed_dict变量）来传入，接下来，我们为x创建一个占位符，这将允许我们在稍后运行会话时传入一个数字。

```python
#利用feed_dict来改变x的值

x = tf.placeholder(tf.int64,name="x")
print(sess.run(2 * x,feed_dict={x:3}))
sess.close()
```

执行结果：

```python
6
```

当我们第一次定义x时，我们不必为它指定一个值。 占位符只是一个变量，我们会在运行会话时将数据分配给它。

#### 1.1 - 线性函数

让我们通过计算以下等式来开始编程：Y=WX+b,W和X是随机矩阵，b是随机向量。 

我们计算WX+b，其中W，X和b是从随机正态分布中抽取的。 W的维度是（4,3），X是（3,1），b是（4,1）。 我们开始定义一个shape=（3,1）的常量X：

X = tf.constant(np.random.randn(3,1), name = "X")

```python
def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b 

    """

    np.random.seed(1) #指定随机种子

    X = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W,X),b) #tf.matmul是矩阵乘法
    #Y = tf.matmul(W,X) + b #也可以以写成这样子

    #创建一个session并运行它
    sess = tf.Session()
    result = sess.run(Y)

    #session使用完毕，关闭它
    sess.close()

    return result
```

我们来测试一下：

```python
print("result = " +  str(linear_function()))
```

测试结果：

```python
result = [[-2.15657382]
 [ 2.95891446]
 [-1.08926781]
 [-0.84538042]]
```

#### 1.2 - 计算sigmoid

我们已经实现了线性函数，TensorFlow提供了多种常用的神经网络的函数比如tf.softmax和 tf.sigmoid。

我们将使用占位符变量x，当运行这个session的时候，我们西药使用使用feed字典来输入z，我们将创建占位符变量x，使用tf.sigmoid来定义操作符，最后运行session，我们会用到下面的代码：

- tf.placeholder(tf.float32, name = “…”)
- tf.sigmoid(…)
- sess.run(…, feed_dict = {x: z})

需要注意的是我们可以使用两种方法来创建并使用session

**方法一**：

```python
sess = tf.Session()
result = sess.run(...,feed_dict = {...})
sess.close()
```

**方法二**：

```python
with tf.Session as sess:
    result = sess.run(...,feed_dict = {...})
```

我们来实现它：

```python
def sigmoid(z):
    """
    实现使用sigmoid函数计算z

    参数：
        z - 输入的值，标量或矢量

    返回：
        result - 用sigmoid计算z的值

    """

    #创建一个占位符x，名字叫“x”
    x = tf.placeholder(tf.float32,name="x")

    #计算sigmoid(z)
    sigmoid = tf.sigmoid(x)

    #创建一个会话，使用方法二
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})

    return result
```

现在我们测试一下：

```python
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
```

测试结果：

```python
sigmoid(0) = 0.5
sigmoid(12) = 0.999994
```

#### 1.3 - 计算成本

还可以使用内置函数计算神经网络的成本。因此，不需要编写代码来计算成本函数的 a2(i) 和 y(i)for i=1…m。

实现成本函数，需要用到的是： 
tf.nn.sigmoid_cross_entropy_with_logits(logits = ..., labels = ...)

你的代码应该输入z，计算sigmoid（得到 a），然后计算交叉熵成本J，所有的步骤都可以通过一次调用tf.nn.sigmoid_cross_entropy_with_logits来完成。

#### 1.4 - 使用独热编码（0、1编码）

很多时候在深度学习中y向量的维度是从0到C−1的，C是指分类的类别数量，如果C=4，那么对y而言你可能需要有以下的转换方式: 

![onehot](https://img-blog.csdn.net/20180417110137252?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这叫做独热编码（”one hot” encoding），因为在转换后的表示中，每列的一个元素是“hot”（意思是设置为1）。 要在numpy中进行这种转换，您可能需要编写几行代码。 在tensorflow中，只需要使用一行代码：

```python
tf.one_hot(labels,depth,axis)
```

下面我们要做的是取一个标签矢量和C类总数，返回一个独热编码。

```python
def one_hot_matrix(lables,C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1

    参数：
        lables - 标签向量
        C - 分类数

    返回：
        one_hot - 独热矩阵

    """

    #创建一个tf.constant，赋值为C，名字叫C
    C = tf.constant(C,name="C")

    #使用tf.one_hot，注意一下axis
    one_hot_matrix = tf.one_hot(indices=lables , depth=C , axis=0)

    #创建一个session
    sess = tf.Session()

    #运行session
    one_hot = sess.run(one_hot_matrix)

    #关闭session
    sess.close()

    return one_hot
```

现在我们来测试一下：

```python
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels,C=4)
print(str(one_hot))
```

测试结果：

```python
[[ 0.  0.  0.  1.  0.  0.]
 [ 1.  0.  0.  0.  0.  1.]
 [ 0.  1.  0.  0.  1.  0.]
 [ 0.  0.  1.  0.  0.  0.]]
```

#### 1.5 - 初始化为0和1

现在我们将学习如何用0或者1初始化一个向量，我们要用到`tf.ones()`和`tf.zeros()`，给定这些函数一个维度值那么它们将会返回全是1或0的满足条件的向量/矩阵，我们来看看怎样实现它们：

```python
def ones(shape):
    """
    创建一个维度为shape的变量，其值全为1

    参数：
        shape - 你要创建的数组的维度

    返回：
        ones - 只包含1的数组    
    """

    #使用tf.ones()
    ones = tf.ones(shape)

    #创建会话
    sess = tf.Session()

    #运行会话
    ones = sess.run(ones)

    #关闭会话
    sess.close()

    return ones
```

测试一下：

```python
print ("ones = " + str(ones([3])))
```

测试结果：

```python
ones = [ 1.  1.  1.]
```

#### 2 - 使用TensorFlow构建你的第一个神经网络

我们将会使用TensorFlow构建一个神经网络，需要记住的是实现模型需要做以下两个步骤： 

1. 创建计算图
2. 运行计算图

我们开始一步步地走一下：

#### 2.0 - 要解决的问题

一天下午，我们和一些朋友决定教我们的电脑破译手语。我们花了几个小时在白色的墙壁前拍照，于是就有了了以下数据集。现在，你的任务是建立一个算法，使有语音障碍的人与不懂手语的人交流。

- 训练集：有从0到5的数字的1080张图片(64x64像素)，每个数字拥有180张图片。
- 测试集：有从0到5的数字的120张图片(64x64像素)，每个数字拥有5张图片。

需要注意的是这是完整数据集的一个子集，完整的数据集包含更多的符号。

下面是每个数字的样本，以及我们如何表示标签的解释。这些都是原始图片，我们实际上用的是64 * 64像素的图片。

![hands](https://img-blog.csdn.net/20180417110431384?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

首先我们需要加载数据集：

```python
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = tf_utils.load_dataset()
```

我们可以看一下数据集里面有什么，当然你也可以自己更改一下index的值。

```python
index = 11
plt.imshow(X_train_orig[index])
print("Y = " + str(np.squeeze(Y_train_orig[:,index])))
```

执行结果：

```python
Y = 1
```

![y=1](https://img-blog.csdn.net/20180417110635257?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

和往常一样，我们要对数据集进行扁平化，然后再除以255以归一化数据，除此之外，我们要需要把每个标签转化为独热向量，像上面的图一样。

```python
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T #每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

print("训练集样本数 = " + str(X_train.shape[1]))
print("测试集样本数 = " + str(X_test.shape[1]))
print("X_train.shape: " + str(X_train.shape))
print("Y_train.shape: " + str(Y_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("Y_test.shape: " + str(Y_test.shape))
```

执行结果：

```python
训练集样本数 = 1080
测试集样本数 = 120
X_train.shape: (12288, 1080)
Y_train.shape: (6, 1080)
X_test.shape: (12288, 120)
Y_test.shape: (6, 120)
```

我们的目标是构建能够高准确度识别符号的算法。 要做到这一点，你要建立一个TensorFlow模型，这个模型几乎和你之前在猫识别中使用的numpy一样（但现在使用softmax输出）。要将您的numpy实现与tensorflow实现进行比较的话这是一个很好的机会。

目前的模型是：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX，SIGMOID输出层已经转换为SOFTMAX。当有两个以上的类时，一个SOFTMAX层将SIGMOID一般化。

#### 2.1 - 创建placeholders

我们的第一项任务是为X和Y创建占位符，这将允许我们稍后在运行会话时传递您的训练数据。

```python
def create_placeholders(n_x,n_y):
    """
    为TensorFlow会话创建占位符
    参数：
        n_x - 一个实数，图片向量的大小（64*64*3 = 12288）
        n_y - 一个实数，分类数（从0到5，所以n_y = 6）

    返回：
        X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y - 一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"

    提示：
        使用None，因为它让我们可以灵活处理占位符提供的样本数量。事实上，测试/训练期间的样本数量是不同的。

    """

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y
```

测试一下：

```python
X, Y = create_placeholders(12288, 6)
print("X = " + str(X))
print("Y = " + str(Y))
```

测试结果：

```python
X = Tensor("X:0", shape=(12288, ?), dtype=float32)
Y = Tensor("Y:0", shape=(6, ?), dtype=float32)
```

#### 2.2 - 初始化参数

初始化tensorflow中的参数，我们将使用Xavier初始化权重和用零来初始化偏差，比如：

```python
W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
```

注：`tf.Variable()` 每次都在创建新对象，对于`get_variable()`来说，对于已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。

```python
def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    返回：
        parameters - 包含了W和b的字典


    """

    tf.set_random_seed(1) #指定随机种子

    W1 = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters
```

测试一下：

```python
tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。 

with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
```

测试结果：

```python
W1 = <tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>
b1 = <tf.Variable 'b1:0' shape=(25, 1) dtype=float32_ref>
W2 = <tf.Variable 'W2:0' shape=(12, 25) dtype=float32_ref>
b2 = <tf.Variable 'b2:0' shape=(12, 1) dtype=float32_ref>
```

正如预期的那样，这些参数只有物理空间，但是还没有被赋值，这是因为没有通过session执行。

#### 2.3 - 前向传播

我们将要在TensorFlow中实现前向传播，该函数将接受一个字典参数并完成前向传播，它会用到以下代码：

- tf.add(…) ：加法
- tf.matmul(… , …) ：矩阵乘法
- tf.nn.relu(…) ：Relu激活函数

我们要实现神经网络的前向传播，我们会拿numpy与TensorFlow实现的神经网络的代码作比较。最重要的是前向传播要在Z3处停止，因为在TensorFlow中最后的线性输出层的输出作为计算损失函数的输入，所以不需要A3.

```python
def forward_propagation(X,parameters):
    """
    实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    参数：
        X - 输入数据的占位符，维度为（输入节点数量，样本数量）
        parameters - 包含了W和b的参数的字典

    返回：
        Z3 - 最后一个LINEAR节点的输出

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)        # Z1 = np.dot(W1, X) + b1
    #Z1 = tf.matmul(W1,X) + b1             #也可以这样写
    A1 = tf.nn.relu(Z1)                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)     # Z3 = np.dot(W3,Z2) + b3


    return Z3
```

测试一下：

```python
tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。 
with tf.Session() as sess:
    X,Y = create_placeholders(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    print("Z3 = " + str(Z3))
```

测试结果：

```python
Z3 = Tensor("Add_2:0", shape=(6, ?), dtype=float32)
```

您可能已经注意到前向传播不会输出任何cache，当我们完成反向传播的时候你就会明白了。

#### 2.4 - 计算成本

如前所述，成本很容易计算：

```python
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
```

我们现在就来实现计算成本的函数：

```
def compute_cost(Z3,Y):
    """
    计算成本

    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个占位符，和Z3的维度相同

    返回：
        cost - 成本值


    """
    logits = tf.transpose(Z3) #转置
    labels = tf.transpose(Y)  #转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost
```

测试一下：

```python
tf.reset_default_graph()

with tf.Session() as sess:
    X,Y = create_placeholders(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)
    print("cost = " + str(cost))
```

测试结果：

```python
cost = Tensor("Mean:0", shape=(), dtype=float32)
```

#### 2.5 - 反向传播&更新参数

得益于编程框架，所有反向传播和参数更新都在1行代码中处理。计算成本函数后，将创建一个“optimizer”对象。 运行tf.session时，必须将此对象与成本函数一起调用，当被调用时，它将使用所选择的方法和学习速率对给定成本进行优化。

举个例子，对于梯度下降：

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
```

要进行优化，应该这样做：

```python
_ , c = sess.run([optimizer,cost],feed_dict={X:mini_batch_X,Y:mini_batch_Y})
```

编写代码时，我们经常使用 `_` 作为一次性变量来存储我们稍后不需要使用的值。 这里，`_`具有我们不需要的优化器的评估值（并且c取值为成本变量的值）。

#### 2.6 - 构建模型

现在我们将实现我们的模型

```python
def model(X_train,Y_train,X_test,Y_test,
        learning_rate=0.0001,num_epochs=1500,minibatch_size=32,
        print_cost=True,is_plot=True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    参数：
        X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
        Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
        X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
        Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        mini_batch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图

    返回：
        parameters - 学习后的参数

    """
    ops.reset_default_graph()                #能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x , m)  = X_train.shape               #获取输入节点数量和样本数
    n_y = Y_train.shape[0]                   #获取输出节点数量
    costs = []                               #成本集

    #给X和Y创建placeholder
    X,Y = create_placeholders(n_x,n_y)

    #初始化参数
    parameters = initialize_parameters()

    #前向传播
    Z3 = forward_propagation(X,parameters)

    #计算成本
    cost = compute_cost(Z3,Y)

    #反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #初始化所有的变量
    init = tf.global_variables_initializer()

    #开始会话并计算
    with tf.Session() as sess:
        #初始化
        sess.run(init)

        #正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  #每代的成本
            num_minibatches = int(m / minibatch_size)    #minibatch的总数量
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:

                #选择一个minibatch
                (minibatch_X,minibatch_Y) = minibatch

                #数据已经准备好了，开始运行session
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                #计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            #记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                #是否打印：
                if print_cost and epoch % 100 == 0:
                        print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        #是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        #保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        #计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
```

我们来正式运行一下模型，请注意，这次的运行时间大约在5-8分钟左右，如果在`epoch = 100`的时候，你的`epoch_cost = 1.01645776539`的值和我相差过大，那么你就立即停止，回头检查一下哪里出了问题。

```python
#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )
```

运行结果：

```python
epoch = 0    epoch_cost = 1.85570189447
epoch = 100    epoch_cost = 1.01645776539
epoch = 200    epoch_cost = 0.733102379423
epoch = 300    epoch_cost = 0.572938936226
epoch = 400    epoch_cost = 0.468773578604
epoch = 500    epoch_cost = 0.3810211113
epoch = 600    epoch_cost = 0.313826778621
epoch = 700    epoch_cost = 0.254280460603
epoch = 800    epoch_cost = 0.203799342567
epoch = 900    epoch_cost = 0.166511993291
epoch = 1000    epoch_cost = 0.140936921718
epoch = 1100    epoch_cost = 0.107750129745
epoch = 1200    epoch_cost = 0.0862994250475
epoch = 1300    epoch_cost = 0.0609485416137
epoch = 1400    epoch_cost = 0.0509344103436
参数已经保存到session。
训练集的准确率： 0.999074
测试集的准确率: 0.725
CPU的执行时间 = 482.19651398680486 秒
```

![plot](https://img-blog.csdn.net/20180417111935425?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

现在，我们的算法已经可以识别0-5的手势符号了，准确率在72.5%。 

我们的模型看起来足够大了，可以适应训练集，但是考虑到训练与测试的差异，你也完全可以尝试添加L2或者dropout来减少过拟合。将session视为一组代码来训练模型，在每个minibatch上运行会话时，都会训练我们的参数，总的来说，你已经运行了很多次（1500代），直到你获得训练有素的参数。

### 相关库代码

#### tf_utils.py

```python
#tf_utils.py

import h5py
import numpy as np
import tensorflow as tf
import math

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    return Z3
```


