# iLab暑期实习

* [实习内容](#实习内容)
* [仓库结构](#仓库结构)
* [关于作者](#关于作者)

-----

## 实习内容

- 学习笔记
  - [7.20学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
  - [7.21学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-21%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
  - [7.22学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-22%E5%BC%A0%E5%96%86%E7%AC%94%E8%AE%B0.md)
  - [7.23学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-23%E5%BC%A0%E5%96%86%E7%AC%94%E8%AE%B0.md)
  - [7.24学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-24%E5%BC%A0%E5%96%86%E7%AC%94%E8%AE%B0.md)
  - [7.25学习笔记](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/7-25%E5%BC%A0%E5%96%86%E7%AC%94%E8%AE%B0.md)
  - [常见的网络结构](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Learning%20Notes/%E5%B8%B8%E8%A7%81%E7%9A%84%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.md)
- [实验笔记](https://github.com/doubleZ0108/iLab-SummerResearch/tree/master/Jupyter%20Notebooks)
- 课后作业
  - [第一周测验-深度学习简介](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Course%20Test/%E7%AC%AC1%E5%91%A8%E6%B5%8B%E9%AA%8C%20-%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
  - [第二周测验-神经网络基础](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Course%20Test/%E7%AC%AC2%E5%91%A8%E6%B5%8B%E9%AA%8C%20-%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80.md)
- [编程作业](https://github.com/doubleZ0108/iLab-SummerResearch/tree/master/Coding%20Test/Weak1%262)
- [MNIST手写数字识别综合项目](https://github.com/doubleZ0108/iLab-SummerResearch/tree/master/MNIST)
- 学习资料
  - [Deep learning深度学习笔记v5.47](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/Resources/Deeplearning%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0v5.47.pdf)
- 相关文档
  - [暑假iLab实验室实习报告](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/doc/17%E6%9C%AC-%E5%BC%A0%E5%96%86-1754060-%E5%90%8C%E6%B5%8E%E5%A4%A7%E5%AD%A6ilab%E5%AE%9E%E9%AA%8C%E5%AE%A4.pdf)
  - [同济大学软件学院学生实习鉴定表](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/doc/%E5%90%8C%E6%B5%8E%E5%A4%A7%E5%AD%A6%E8%BD%AF%E4%BB%B6%E5%AD%A6%E9%99%A2%E5%AD%A6%E7%94%9F%E5%AE%9E%E4%B9%A0%E9%89%B4%E5%AE%9A%E8%A1%A8.doc)
  - [校内实习证明](https://github.com/doubleZ0108/iLab-SummerResearch/blob/master/doc/%E6%A0%A1%E5%86%85%E5%AE%9E%E4%B9%A0%E8%AF%81%E6%98%8E.doc)
  
<br/>

## 仓库结构

```
│  README.md   
│          
├─── Coding Test				 //编程作业💻   
│   └── Weak1&2    
│       ├── datasets    
│       │   ├── test_signs.h5    
│       │   └── train_signs.h5    
│       ├── notebook    
│       │   ├── Course1_Week3.ipynb    
│       │   ├── Course1_Week4.ipynb    
│       │   ├── Course2_Week1.ipynb    
│       │   ├── Course2_Week2.ipynb    
│       │   ├── Course2_Week3.ipynb    
│       │   ├── Course4_Week1.ipynb    
│       │   ├── hello_tensorflow.ipynb    
│       │   ├── mnist.ipynb    
│       │   └── mnist_cnn.ipynb    
│       ├── src    
│       │   ├── cnn_utils.py    
│       │   ├── dnn_utils.py    
│       │   ├── gc_utils.py    
│       │   ├── init_utils.py    
│       │   ├── lr_utils.py    
│       │   ├── opt_utils.py    
│       │   ├── planar_utils.py    
│       │   ├── reg_utils.py    
│       │   ├── testCase.py    
│       │   ├── testCases.py    
│       │   └── tf_utils.py    
│       └── 吴恩达编程作业_Weak1&2.md    

├─Course Test					//课后作业📒   
│      第1周测验 - 深度学习简介.md    
│      第2周测验 - 神经网络基础.md    
│          
├─Jupyter Notebooks	 			//Jupyter实验笔记🧪    
│      myfirst.ipynb    
│      reshape.ipynb    
│          
├─Learning Notes				//学习笔记✏️  
│      7-20学习笔记.md    
│      7-21学习笔记.md    
│      7-22张喆笔记.md    
│      7-23张喆笔记.md    
│      7-24张喆笔记.md    
│      7-25张喆笔记.md    
│      常见的网络结构.md	  			 ...LeNet, AlexNet, Googlenet, VGG, Resent 网络结构学习笔记    
│          
├─MNIST						//MNIST数据集综合项目:slot_machine:    
│  │  README.md 					...项目说明文档   
│  │  对手写数字识别的简单优化.md				...优化算法文档    
│  │      
│  ├─datasets    
│  │      mnist.npz    
│  │          
│  ├─notebook    
│  │      minst_cnn.ipynb				...卷积算法实验笔记    
│  │      minst_mlp.ipynb				...多层神经网络实验笔记    
│  │          
│  └─src    
│          mnist_cnn.py					...卷积神经网络代码    
│          mnist_mlp.py					...多层神经网络代码    
│          optimize.py					...优化算法代码    
│              
└─Resources					//学习资料💾    
        Deeplearning深度学习笔记v5.47.pdf    
            

```

<br/>

## 关于作者

| 姓名                    | 学号    |
| ----------------------- | ------- |
| 张喆<sup>「组长」</sup> | 1754060 |
| 卜滴                    | 1753414 |
| 刘一默                  | 1752339 |
