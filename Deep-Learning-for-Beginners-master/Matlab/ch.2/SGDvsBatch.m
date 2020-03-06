% 该文件将随机梯度下降（SGD）方法与 batch 化随机梯度下降（batch-SGD）方法进行了性能比较。
% 由比较结果可以得知，相比于 batch SGD 而言，SGD 使用更少的迭代次数，就可以达到更低的错误率。

clear all

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      0
      1
      1
    ];


E1 = zeros(1000, 1); % SGD 方法中每个 epoch 的损失函数值统计
                     % 1000x1 矩阵（1000 维向量）
E2 = zeros(1000, 1); % batch-SGD 方法中每个 epoch 的损失函数值统计
                     % 1000x1 矩阵（1000 维向量）

W1 = 2*rand(1, 3) - 1; % 随机初始化权重参数
                       % 每个元素的值在 -1 到 1 之间
W2 = W1; % 这是一次对比实验，因而我们需要保证权重参数的初始值是相同的

% 训练模型
% 注意这里仅用了 1000 次迭代（之前是 10000 次），因为 1000 次迭代就足以看出性能差异
for epoch = 1:1000
  W1 = DeltaSGD(W1, X, D);
  W2 = DeltaBatch(W2, X, D);

  es1 = 0;
  es2 = 0;
  N   = 4;
  for k = 1:N
    x = X(k, :)';
    d = D(k);

    v1  = W1*x;
    y1  = Sigmoid(v1);
    es1 = es1 + (d - y1)^2; % L2 损失函数

    v2  = W2*x;
    y2  = Sigmoid(v2);
    es2 = es2 + (d - y2)^2;
  end
  E1(epoch) = es1 / N;
  E2(epoch) = es2 / N;
end

plot(E1, 'r')
hold on
plot(E2, 'b:')
xlabel('Epoch')
ylabel('Average of Training error')
legend('SGD', 'Batch')

