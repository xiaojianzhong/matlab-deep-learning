# 第四章

## 目录说明

- [MultiClass.m](./MultiClass.m) 展示了如何使用多层神经网络解决多类别分类问题。
- [TestMultiClass.m](./TestMultiClass.m) 测试了 [MultiClass.m](./MultiClass.m)。
- [RealMultiClass.m](./RealMultiClass.m) 使用真实数据测试了 [MultiClass.m](./MultiClass.m)。
- [rng.m](./rng.m) 定义了 `rng` 函数。
- [Sigmoid.m](./Sigmoid.m) 定义了 `Sigmoid` 函数。
- [Softmax.m](./Softmax.m) 定义了 `Softmax` 函数。

## 网络结构

| 文件                           | 更新策略 | 输入层            | 隐藏层                                                 | 输出层                                                                   |
| ------------------------------ | -------- | ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------------ |
| [MultiClass.m](./MultiClass.m) | SGD      | 节点数：25（5x5） | 节点数：50<br/>激活函数：sigmoid<br/>损失函数：L2 loss | 节点数：5<br/>激活函数：**softmax**<br/>损失函数：**cross entropy loss** |
