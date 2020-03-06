# 第三章

## 目录说明

- [BackpropXOR.m](./BackpropXOR.m) 文件展示了如何通过反向传播算法更新多层神经网络的权重参数。
- [TestBackpropXOR.m](./TestBackpropXOR.m) 测试了 [BackpropXOR.m](./BackpropXOR.m)。
- [BackpropMmt.m](./BackpropMmt.m) 文件展示了如何结合动量理论来提高参数更新过程中的稳定性。
- [TestBackpropMmt.m](./TestBackpropMmt.m) 测试了 [BackpropMmt.m](./BackpropMmt.m)。
- [BackpropCE.m](./BackpropCE.m) 文件展示了如何在输出层使用 cross entropy 损失函数来加快收敛速度。
- [TestBackpropCE.m](./TestBackpropCE.m) 测试了 [BackpropCE.m](./BackpropCE.m)。
- [CEvsSSE.m](./CEvsSSE.m) 比较了 cross entropy 损失函数与 L2 损失函数的性能表现。
- [Sigmoid.m](./Sigmoid.m) 定义了 `Sigmoid` 函数。

## 网络结构

| 文件                                                                  | 更新策略 | 输入层    | 隐藏层                                                | 输出层                                                               |
| --------------------------------------------------------------------- | -------- | --------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| [BackpropXOR.m](./BackpropXOR.m)<br/>[BackpropMmt.m](./BackpropMmt.m) | SGD      | 节点数：3 | 节点数：4<br/>激活函数：sigmoid<br/>损失函数：L2 loss | 节点数：1<br/>激活函数：sigmoid<br/>损失函数：**L2 loss**            |
| [BackpropCE.m](./BackpropCE.m)                                        | SGD      | 节点数：3 | 节点数：4<br/>激活函数：sigmoid<br/>损失函数：L2 loss | 节点数：1<br/>激活函数：sigmoid<br/>损失函数：**cross entropy loss** |
