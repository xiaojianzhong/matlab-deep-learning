# 第二章

## 目录说明

- [DeltaSGD.m](./DeltaSGD.m) 展示了如何通过随机梯度下降方法更新权重参数。
- [TestDeltaSGD.m](./TestDeltaSGD.m) 测试了 [DeltaSGD.m](./DeltaSGD.m)。
- [DeltaBatch.m](./DeltaBatch.m) 展示了如何通过 batch 化随机梯度下降方法更新权重参数。
- [TestDeltaBatch.m](./TestDeltaBatch.m) 测试了 [DeltaBatch.m](./DeltaBatch.m)。
- [DeltaXOR.m](./DeltaXOR.m) 与 [DeltaSGD.m](./DeltaSGD.m) 内容相同。
- [TestDeltaSGD.m](./TestDeltaSGD.m) 说明了为何 single-layer neural network 不适用于非线性可解的问题（这里用 XOR 问题说明）。
- [SGDvsBatch.m](./SGDvsBatch.m) 比较了随机梯度下降方法与 batch 化随机梯度下降方法的性能表现。
- [Sigmoid.m](./Sigmoid.m) 定义了 `Sigmoid` 函数。

## 网络结构

| 文件                                                      | 更新策略      | 输入层    | 输出层                                                |
| --------------------------------------------------------- | ------------- | --------- | ----------------------------------------------------- |
| [DeltaSGD.m](./DeltaSGD.m)<br/>[DeltaXOR.m](./DeltaXOR.m) | **SGD**       | 节点数：3 | 节点数：1<br/>激活函数：sigmoid<br/>损失函数：L2 loss |
| [DeltaBatch.m](./DeltaBatch.m)<br/>                       | **batch-SGD** | 节点数：3 | 节点数：1<br/>激活函数：sigmoid<br/>损失函数：L2 loss |
