% 该文件对 BackpropCE 函数进行了测试。

clear all

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      1
      1
      0
    ]; % XOR 问题

W1 = 2*rand(4, 3) - 1; % 随机初始化权重参数
                       % 每个元素的值在 -1 到 1 之间
W2 = 2*rand(1, 4) - 1; % 随机初始化权重参数
                       % 每个元素的值在 -1 到 1 之间

% 训练模型
for epoch = 1:10000
  [W1 W2] = BackpropCE(W1, W2, X, D);
end

% 使用模型进行预测
N = 4;
for k = 1:N
  x  = X(k, :)';
  v1 = W1*x;
  y1 = Sigmoid(v1);
  v  = W2*y1;
  y  = Sigmoid(v)
end
