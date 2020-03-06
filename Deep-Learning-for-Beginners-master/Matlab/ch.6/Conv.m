% Conv 函数对输入矩阵用指定的过滤器集合执行步长为 1 的卷积操作。
% 卷积层是一种特殊的神经网络层，它不同于一般的加权和。关于卷积层的更多信息，可以参考 https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer。
% 注意该函数并不适用于步长为其他值的情况。
%
% x 是一个矩阵，代表要进行卷积的图像。
% W 是一个 3 维度张量，是卷积操作所要使用到的过滤器的集合。
%
% 该函数会返回一个 3 维度张量 y，代表卷积结果 feature map。
% 根据卷积操作的理论我们可以得知，该返回张量 y 的形状满足一定的数量关系。
function y = Conv(x, W)
  [wrow, wcol, numFilters] = size(W); % wrow，wcol 分别为过滤器的行数和列数，numFilters 为过滤器的数量
  [xrow, xcol, ~         ] = size(x); % xrow，xcol 分别为图像的行数和列数

  yrow = xrow - wrow + 1; % yrow 为 feature map 的行数，由卷积理论可知其值为：(xrow - wrow) / step + 1，在这里 step 的值为 1
  ycol = xcol - wcol + 1; % ycol 为 feature map 的列数，由卷积理论可知其值为：(xcol - wcol) / step + 1，在这里 step 的值为 1

  y = zeros(yrow, ycol, numFilters); % feature map 第三维的大小应该等于过滤器的数量

  for k = 1:numFilters
    filter = W(:, :, k); % 取出第 k 个过滤器
    filter = rot90(squeeze(filter), 2); % ？
                                        % squeeze 是 Matlab 内置函数，它会删除掉张量中长度为 1 的维度，详见 https://ww2.mathworks.cn/help/matlab/ref/squeeze.html
                                        % rot90 是 Matlab 内置函数，它会将张量逆时针旋转指定次数，详见 https://ww2.mathworks.cn/help/matlab/ref/rot90.html
    y(:, :, k) = conv2(x, filter, 'valid'); % 将图像与所取出的过滤器进行卷积，结果保存在 feature map 中
                                            % conv2 是 Matlab 内置函数，它执行两个矩阵的二维卷积，详见 https://ww2.mathworks.cn/help/matlab/ref/conv2.html
  end
end
