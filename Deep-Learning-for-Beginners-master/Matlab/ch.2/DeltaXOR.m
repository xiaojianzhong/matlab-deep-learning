% DeltaXOR 函数在一次 epoch 中，使用随机梯度下降方法，通过训练数据集对权重进行更新。
% 该函数与 DeltaSGD 函数的内容完全相同。
%
% W 为 1x3 矩阵，是输入层与输出层之间的权重参数。
% X 为 4x3 矩阵，包含所有训练样本的特征。
% D 为 4x1 矩阵（4 维向量），包含所有训练样本的标签。
%
% 该函数返回更新后的权重参数 W。
function W = DeltaXOR(W, X, D)
  alpha = 0.9; % 学习率

  N = 4; % 训练数据的数量，亦即矩阵 X 的行数量
  for k = 1:N
    x = X(k, :)'; % 一个单独的训练样本的特征向量
                  % 注意这里进行了转置，将行向量转换为列向量
                  % 3 维向量
    d = D(k); % 该样本的标签
              % 标量

    v = W*x; % 输出层的加权和
             % 标量（[1x3] x [3x1] = [1x1]）
    y = Sigmoid(v); % 输出层的激活值
                    % 标量

    e     = d - y; % 输出层的 e
                   % 标量
    delta = y*(1-y)*e; % 输出层的 delta，其中 y*(1-y) 是 sigmoid 函数的导数
                       % 标量

    dW = alpha*delta*x; % 权重参数的更新值
                        % 3 维向量

    % 更新权重参数
    % 由于 W 是 1x3 矩阵，而 dW 是 3x1 矩阵，故不能直接执行 W = W + dW
    W(1) = W(1) + dW(1);
    W(2) = W(2) + dW(2);
    W(3) = W(3) + dW(3);
  end
end
