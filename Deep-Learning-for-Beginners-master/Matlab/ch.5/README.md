# 第五章

## 目录说明

- [DeepReLU.m](./DeepReLU.m) 展示了如何使用 ReLU 激活函数来避免梯度消失问题。
- [TestDeepReLU.m](./TestDeepReLU.m) 测试了 [DeepReLU.m](./DeepReLU.m)。
- [DeepDropout.m](./DeepDropout.m) 展示了如何采取 dropout 策略来避免过拟合问题。
- [TestDeepDropout.m](./TestDeepDropout.m) 测试了 [DeepDropout.m](./DeepDropout.m)。
- [Sigmoid.m](./Sigmoid.m) 定义了 `Sigmoid` 函数。
- [Softmax.m](./Softmax.m) 定义了 `Softmax` 函数。
- [ReLU.m](./ReLU.m) 定义了 `ReLU` 函数。
- [Dropout.m](./Dropout.m) 定义了 `Dropout` 函数。

## 网络结构

| 文件                             | 更新策略 | 输入层            | 隐藏层 1                                                | 隐藏层 2                                                | 隐藏层 3                                                | 输出层                                                                   |
| -------------------------------- | -------- | ----------------- | ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| [DeepReLU.m](./DeepReLU.m)       | SGD      | 节点数：25（5x5） | 节点数：20<br/>激活函数：**ReLU**<br/>损失函数：L2 loss | 节点数：20<br/>激活函数：**ReLU**<br/>损失函数：L2 loss | 节点数：20<br/>激活函数：**ReLU**<br/>损失函数：L2 loss | 节点数：5<br/>激活函数：**softmax**<br/>损失函数：**cross entropy loss** |
| [DeepDropout.m](./DeepDropout.m) | SGD      | 节点数：25（5x5） | 节点数：20<br/>激活函数：sigmoid<br/>损失函数：L2 loss  | 节点数：20<br/>激活函数：sigmoid<br/>损失函数：L2 loss  | 节点数：20<br/>激活函数：sigmoid<br/>损失函数：L2 loss  | 节点数：5<br/>激活函数：**softmax**<br/>损失函数：**cross entropy loss** |
