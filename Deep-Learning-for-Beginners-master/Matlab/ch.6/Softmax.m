% Softmax 函数对输入向量执行 softmax 变换。
% softmax 是激活函数的一种，关于 sigmoid 的更多信息，可以参考 https://en.wikipedia.org/wiki/Softmax_function。
%
% x 是一个向量。
% 注意 Softmax 函数的输入必须是一个向量，不能是一个标量值。
%
% 该函数会返回一个大小与向量 x 相同的向量 y。
% 依据 softmax 函数的性质我们可以得知，该返回向量 y 的所有分量之和必定为 1。
function y = Softmax(x)
  ex = exp(x);
  y  = ex / sum(ex);
end
