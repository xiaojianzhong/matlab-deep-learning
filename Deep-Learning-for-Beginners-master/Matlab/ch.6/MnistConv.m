% MnistConv 函数基于 minibatch 化随机梯度下降策略，使用反向传播算法，通过训练数据集对权重进行更新。
% MnistConv 函数使用多层卷积神经网络解决图像识别问题。
%
% W1 为 9x9x20 张量，是输入层与隐藏层 1 之间的权重参数，由于隐藏层 1 是卷积层，所以 W1 实际上是 9x9 大小的过滤器矩阵的集合。
% W5 为 100x2000 矩阵，是隐藏层 4 与隐藏层 5 之间的权重参数。
% Wo 为 10x100 矩阵，是隐藏层 5 与输出层之间的权重参数。
% X 为 28x28x8000 张量，包含所有训练样本的特征。
% D 为 8000x1 矩阵（8000 维向量），包含所有训练样本的标签。
%
% 该函数返回更新后的权重参数 W1、W5 与 Wo。
function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)
  alpha = 0.01; % 学习率
                % 注意这里的值比起之前的模型要低很多
  beta  = 0.95; % 先前动量的影响因子，和 alpha 一样属于超参数

  momentum1 = zeros(size(W1)); % 记录权重 W1 更新过程中的动量
                               % 9x9x20 张量
  momentum5 = zeros(size(W5)); % 记录权重 W5 更新过程中的动量
                               % 100x2000 矩阵
  momentumo = zeros(size(Wo)); % 记录权重 Wo 更新过程中的动量
                               % 10x100 矩阵

  N = length(D); % 训练数据的数量，亦即张量 X 第一维的大小

  bsize = 100; % minibatch-SGD 所使用的一个批次里样本的数量，同为超参数
  blist = 1:bsize:(N-bsize+1); % 保存所有批次中第一个样本的下标的数组
                               % 双冒号是 Matlab 中特有的创建数组的方式，这里创建了一个从 1 开始，到 N-bsize+1 结束的，以 bsize 为单位递增的数组
                               % 详见 https://ww2.mathworks.cn/help/matlab/matlab_prog/matlab-operators-and-special-characters.html

  for batch = 1:length(blist)
    dW1 = zeros(size(W1)); % 权重参数 W1 更新值的和
                           % 9x9x20 张量
    dW5 = zeros(size(W5)); % 权重参数 W5 更新值的和
                           % 100x2000 矩阵
    dWo = zeros(size(Wo)); % 权重参数 Wo 更新值的和
                           % 10x100 矩阵

    begin = blist(batch); % 取出当前批次中第一个样本的下标
    for k = begin:begin+bsize-1
      x  = X(:, :, k); % 一个单独的训练样本的特征向量
                       % 注意这里并没有进行 reshape，而是直接作为图像进行处理
                       % 9x9 矩阵
      y1 = Conv(x, W1); % 隐藏层 1（卷积层）的卷积结果 feature map
                        % 20x20x20 张量
      y2 = ReLU(y1); % 隐藏层 2 的激活值
                     % 20x20x20 张量
      y3 = Pool(y2); % 隐藏层 3（池化层）的池化结果
                     % 10x10x20 张量
      y4 = reshape(y3, [], 1); % 隐藏层 4 的 reshape 结果
                               % 注意该层并没有进行实质性的运算，而只是将张量转换为了向量
                               % 2000 维向量
      v5 = W5*y4; % 隐藏层 5 的加权和
                  % 100x1 矩阵（100 维向量）（[100x2000] x [2000x1] = [100x1]）
      y5 = ReLU(v5); % 隐藏层 5 的激活值
                     % 100x1 矩阵（100 维向量）
      v  = Wo*y5; % 输出层的加权和
                  % 10x1 矩阵（10 维向量）（[10x100] x [100x1] = [10x1]）
      y  = Softmax(v); % 输出层的激活值
                       % 10x1 矩阵（10 维向量）

      d = zeros(10, 1); % 10x1 矩阵（10 维向量）
      d(sub2ind(size(d), D(k), 1)) = 1; % 注意 D 中每一行仅是一个数字标签，我们需要将它转换为 one-hot 编码
                                        % 具体而言，对于一个标签 7，我们需要将它转换为向量 [0,0,0,0,0,0,1,0,0,0]
                                        % sub2ind 是 Matlab 内置函数，它将一个下标转换为一个线性索引，详见 https://ww2.mathworks.cn/help/matlab/ref/sub2ind.html

      % 注意到这里虽然只需要更新 W1，W5 和 Wo，但是需要计算出每一层的 delta 值，这是因为反向传播算法是逐层回溯的，前一层的更新值的计算依赖于后一层
      e      = d - y; % 输出层的 e
                      % 10x1 矩阵（10 维向量）
      delta  = e; % 输出层的 delta
                  % 这里由于同时使用了 softmax 激活函数和 cross entropy 损失函数，所以 delta = e，详见《Matlab Deep Learning》P95
                  % 10x1 矩阵（10 维向量）

      e5     = Wo' * delta; % 隐藏层 5 的 e
                            % 100x1 矩阵（100 维向量）（[10x100]' x [10x1] = [100x1]）
      delta5 = (y5 > 0) .* e5; % 隐藏层 5 的 delta，其中 (y5 > 0) 是 ReLU 函数的导数
                               % 100x1 矩阵（100 维向量）

      e4     = W5' * delta5; % 隐藏层 4 的 e
                             % 2000x1 矩阵（2000 维向量）（[100x2000]' x [100x1] = [2000x1]）

      e3     = reshape(e4, size(y3)); % 隐藏层 3 的 e
                                      % 10x10x20 张量
                                      % 注意隐藏层 3 的 e 其实就是对隐藏层 4 的 e 进行 reshape 得到的

      % 下面对ReLU 层-池化层的损失的反向传播过程的计算超出了本书的范围，不做介绍
      e2 = zeros(size(y2)); % 隐藏层 2 的 e
                            % 20x20x20 张量
      W3 = ones(size(y2)) / (2*2); % 将池化层视为一种特殊的卷积层
      for c = 1:20
        e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
      end

      delta2 = (y2 > 0) .* e2; % 隐藏层 2 的 delta，其中 (y2 > 0) 是 ReLU 函数的导数
                               % 20x20x20 张量

      % 下面对卷积层-ReLU 层的损失的反向传播过程的计算超出了本书的范围，不做介绍
      delta1_x = zeros(size(W1)); % 注意这里的 delta1_x 不是通常意义上的 delta
      for c = 1:20
        delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
      end

      dW1 = dW1 + delta1_x; % 输入层-隐藏层 1 权重参数的更新值
                            % 9x9x20 张量
      dW5 = dW5 + delta5*y4'; % 隐藏层 4-隐藏层 5 权重参数的更新值
                              % 注意这里将 y4 进行了转置，将列向量转换为行向量
                              % 100x2000 矩阵（[100x1] x [2000x1]' = [100x2000]）
      dWo = dWo + delta *y5'; % 隐藏层 5-输出层权重参数的更新值
                              % 注意这里将 y5 进行了转置，将列向量转换为行向量
                              % 10x100 矩阵（[10x1] x [100x1]' = [10x100]）
    end

    dW1 = dW1 / bsize; % 输入层-隐藏层 1 权重参数更新值的平均值
                       % 9x9x20 张量
    dW5 = dW5 / bsize; % 隐藏层 4-隐藏层 5 权重参数更新值的平均值
                       % 100x2000 矩阵
    dWo = dWo / bsize; % 隐藏层 5-输出层权重参数的更新值
                       % 10x100 矩阵

    momentum1 = alpha*dW1 + beta*momentum1; % 更新动量
    W1        = W1 + momentum1; % momentum1 和 W1 的形状相同

    momentum5 = alpha*dW5 + beta*momentum5; % 更新动量
    W5        = W5 + momentum5; % momentum5 和 W5 的形状相同

    momentumo = alpha*dWo + beta*momentumo; % 更新动量
    Wo        = Wo + momentumo; % momentum5 和 W5 的形状相同
  end
end
