% Dropout 函数根据输入向量生成一个屏蔽向量，该屏蔽向量应该以 element-wise 的形式被乘到原始向量上。
% 关于 dropout 的更多信息，可以参考 https://en.wikipedia.org/wiki/Dropout_(neural_networks)。
%
% x 是一个向量。
% ratio 是一个介于 0 到 1 之间的比例标量。
%
% 该函数会返回一个大小与向量 x 相同的向量 y。
% 根据 dropout 策略我们可以得知，该返回向量 y 中有比例为 ratio 的分量被置为 0，其余分量被置为 1 / (1-ratio)。
% 关于为何其余分量需要被置为 1 / (1-ratio)，请参考《Matlab Deep Learning》P117。
function ym = Dropout(y, ratio)
  [m, n] = size(y); % m 为向量 y 的分量数，n 为 1
  ym     = zeros(m, n); % ym 为一个和向量 y 同样大小的、分量全部为 0 的向量

  num     = round(m*n*(1-ratio)); % 不需要被屏蔽的分量的数量
  idx     = randperm(m*n, num); % 不需要被屏蔽的分量的下标
                                % 注意 Matlab 中的下标从 1 开始
                                % randperm 是 Matlab 内置函数，它返回一个随机置换向量，详见 https://ww2.mathworks.cn/help/matlab/ref/randperm.html
  ym(idx) = 1 / (1-ratio); % 将这些分量的值置为 1 / (1-ratio)
end
