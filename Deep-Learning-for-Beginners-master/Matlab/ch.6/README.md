# 第六章

## 目录说明

- [MnistConv.m](./MnistConv.m) 展示了如何使用多层卷积神经网络解决图像识别问题。
- [TestMnistConv.m](./TestMnistConv.m) 测试了 [MnistConv.m](./MnistConv.m)。
- [PlotFeatures.m](./PlotFeatures.m) 展示了在正向传播的过程中图像是如何一步步被处理的。
- [rng.m](./rng.m) 定义了 `rng` 函数。
- [Softmax.m](./Softmax.m) 定义了 `Softmax` 函数。
- [ReLU.m](./ReLU.m) 定义了 `ReLU` 函数。
- [Conv.m](./Conv.m) 定义了 `Conv` 函数。
- [Pool.m](./Pool.m) 定义了 `Pool` 函数。
- [loadMNISTImages.m](./loadMNISTImages.m) 用于从文件系统中加载图片数据，以 3 维度张量的形式返回，来自 [github.com/amaas/stanford_dl_ex/tree/master/common](https://github.com/amaas/stanford_dl_ex/tree/master/common)。不再做具体注释。
- [loadMNISTImages.m](./loadMNISTImages.m) 用于从文件系统中加载标签数据，以向量的形式返回，来自 [github.com/amaas/stanford_dl_ex/tree/master/common](https://github.com/amaas/stanford_dl_ex/tree/master/common)。不再做具体注释。
- [display_network.m](./display_network.m) 用于在屏幕上显示一个矩阵或一个张量，来自 [github.com/amaas/stanford_dl_ex/tree/master/common](https://github.com/amaas/stanford_dl_ex/tree/master/common)。不再做具体注释。
- [MNIST](./MNIST/) 目录下是 MNIST 数据集，来自 [Yann LeCun 的博客](http://yann.lecun.com/exdb/mnist/)。
- [MnistConv.mat](./MnistConv.mat) 是预训练好的模型，通过执行 `load('./MnistConv.mat')` 就可以加载。我对模型进行了重新训练，达到了 94.5% 的准确度。

## 网络结构

| 文件                         | 更新策略      | 输入层               | 隐藏层 1（卷积层）       | 隐藏层 2                                                              | 隐藏层 3（池化层）       | 隐藏层 4     | 隐藏层 5                                                 | 输出层                                                                    |
| ---------------------------- | ------------- | -------------------- | ------------------------ | --------------------------------------------------------------------- | ------------------------ | ------------ | -------------------------------------------------------- | ------------------------------------------------------------------------- |
| [MnistConv.m](./MnistConv.m) | minibatch-SGD | 节点数：784（28x28） | 节点数：8000（20x20x20） | 节点数：8000（20x20x20）<br/>激活函数：**ReLU**<br/>损失函数：L2 loss | 节点数：2000（10x10x20） | 节点数：2000 | 节点数：100<br/>激活函数：**ReLU**<br/>损失函数：L2 loss | 节点数：10<br/>激活函数：**softmax**<br/>损失函数：**cross entropy loss** |
