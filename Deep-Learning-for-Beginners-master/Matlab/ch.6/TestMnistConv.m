% 该文件对 MnistConv 函数进行了测试。

clear all

Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte'); % 从文件系统中加载图片数据
Images = reshape(Images, 28, 28, []); % reshape 为 28x28x10000 的张量
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte'); % 从文件系统中加载标签数据
Labels(Labels == 0) = 10; % 将标签 0 转换为标签 10，以在 one-hot 编码中对应到一个存在的下标（Matlab 的下标从 1 开始，故下标 0 是不存在的）
                          % 注意标签的具体值为多少对神经网络不会造成任何影响，因为每个 one-hot 编码都是等价的，不存在任何地位上的区别

rng(1); % 设置随机数发生器的种子

W1 = 1e-2*randn([9 9 20]); % 随机初始化权重参数
                           % 每个元素的值没有范围限制，但是是从标准正态分布中取出的
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000); % 随机初始化权重参数
Wo = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100); % 随机初始化权重参数

% 取全体数据的前 8000 个样本作为训练集数据
X = Images(:, :, 1:8000);
D = Labels(1:8000);

% 训练模型
% 注意这里仅使用了 3 次迭代（之前是 10000 次）
for epoch = 1:3
  epoch
  [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

% 将训练结果保存到一个 workspace 文件中，下次便可以直接加载，而不需要再次训练
save('MnistConv.mat');


% 取全体数据的后 2000 个样本作为测试集数据
X = Images(:, :, 8001:10000);
D = Labels(8001:10000);

% 使用模型在测试集上对性能进行评估
acc = 0;
N   = length(D);
for k = 1:N
  x = X(:, :, k);

  y1 = Conv(x, W1);
  y2 = ReLU(y1);
  y3 = Pool(y2);
  y4 = reshape(y3, [], 1);
  v5 = W5*y4;
  y5 = ReLU(v5);
  v  = Wo*y5;
  y  = Softmax(v);

  [~, i] = max(y); % 将输出向量再转换回数字标签
  % 这里计算准确度的方法是简单地统计正确预测的标签数量
  if i == D(k)
    acc = acc + 1;
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);


