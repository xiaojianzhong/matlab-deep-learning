% Sigmoid 函数对输入参数执行 sigmoid 变换。
% sigmoid 是激活函数的一种，关于 sigmoid 的更多信息，可以参考 https://en.wikipedia.org/wiki/Sigmoid_function。
%
% x 是一个标量。
%
% 该函数会返回执行 sigmoid 变换后的标量值 y。
function y = Sigmoid(x)
  y = 1 / (1 + exp(-x));
end
