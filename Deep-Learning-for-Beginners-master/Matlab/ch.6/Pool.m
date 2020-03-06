% Pool 函数对输入张量执行 2x2 平均池化（mean pooling）操作。
% 池化层是一种特殊的神经网络层，也是一种特殊的卷积层。关于的更多信息，可以参考 https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer。
% 注意该函数并不适用于其他类型的池化的情况。
%
% x 是一个 3 维度张量，代表要进行池化的 feature map。
%
% 该函数会返回一个 3 维度张量 y，代表池化结果。
% 根据池化操作的理论我们可以得知，该返回张量 y 的形状满足一定的数量关系。
function y = Pool(x)
  [xrow, xcol, numFilters] = size(x); % xrow，xcol 分别为 feature map 的行数和列数，numFilters 为先前卷积用到的过滤器的数量

  y = zeros(xrow/2, xcol/2, numFilters); % 池化结果的行数应该为：feature map 的行数 / 池化器的行数（这里为 2）
                                         % 池化结果的列数应该为：feature map 的列数 / 池化器的列数（这里为 2）
                                         % 池化结果第三维的大小应该等于 feature map 第三维的大小
  for k = 1:numFilters
    filter = ones(2) / (2*2); % 池化操作的等价过滤器
    image  = conv2(x(:, :, k), filter, 'valid'); % 注意这里使用卷积操作完成了池化，因为池化可以视为一种特殊的卷积操作
                                                 % 在这里，“图像进行 2x2 平均池化”等价于“图像与一个 2x2 过滤器（其中每个分量的值为 0.25）进行步长为 2 的卷积”

    y(:, :, k) = image(1:2:end, 1:2:end); % 由于 Matlab 内置的 conv2 函数只支持步长为 1 的卷积操作，所以我们需要手动取出其中步长为 2 的各个像素
  end
end
