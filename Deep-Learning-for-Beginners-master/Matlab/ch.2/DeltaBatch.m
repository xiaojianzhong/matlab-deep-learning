% DeltaBatch 函数在一次 epoch 中，使用 batch 化随机梯度下降方法，通过训练数据集对权重进行更新。
%
% W 为 1x3 矩阵，是输入层与输出层之间的权重参数。
% X 为 4x3 矩阵，包含所有训练样本的特征。
% D 为 4x1 矩阵（4 维向量），包含所有训练样本的标签。
%
% 该函数返回更新后的权重参数 W。
function W = DeltaBatch(W, X, D)
  alpha = 0.9; % 学习率

  dWsum = zeros(3, 1); % 权重参数更新值的和
                       % 3x1 矩阵（3 维向量）

  N = 4; % 训练数据的数量，亦即矩阵 X 的行数量
  for k = 1:N
    x = X(k, :)'; % 一个单独的训练样本的特征向量
                  % 3 维向量
    d = D(k); % 该样本的标签
              % 标量

    v = W*x; % 输出层的加权和
             % 标量（[1x3] x [3x1] = [1x1]）
    y = Sigmoid(v); % 输出层的激活值
                    % 标量

    e     = d - y; % 输出层的损失值
                   % 标量
    delta = y*(1-y)*e; % 输出层的 delta，其中 y*(1-y) 是 sigmoid 函数的导数
                       % 标量

    dW = alpha*delta*x; % 权重参数的更新值
                        % 3 维向量（和权重参数相同）

    dWsum = dWsum + dW;
  end
  dWavg = dWsum / N; % 权重参数更新值的平均值
                     % 3x1 矩阵（3 维向量）

  % 使用平均值来更新权重参数
  % 这里实际上可以直接使用 W = W + dWavg
  W(1) = W(1) + dWavg(1);
  W(2) = W(2) + dWavg(2);
  W(3) = W(3) + dWavg(3);
end
