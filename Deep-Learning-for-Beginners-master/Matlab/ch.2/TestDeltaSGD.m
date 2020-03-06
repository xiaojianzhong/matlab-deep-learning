% 该文件对 DeltaSGD 函数进行了测试。

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

W = 2*rand(1, 3) - 1; % 随机初始化权重参数
                      % 每个元素的值在 -1 到 1 之间

% 训练模型
for epoch = 1:10000
  W = DeltaSGD(W, X, D);
end

% 使用模型进行预测
N = 4;
for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v)
end
