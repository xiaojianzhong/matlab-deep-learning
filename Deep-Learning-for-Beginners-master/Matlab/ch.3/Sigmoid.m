% Sigmoid 函数对输入参数中的每一个元素执行 sigmoid 变换。
% sigmoid 是激活函数的一种，关于 sigmoid 的更多信息，可以参考 https://baike.baidu.com/item/Sigmoid%E5%87%BD%E6%95%B0/7981407。
%
% x 是通用参数，这意味着你可以传入一个标量、向量、矩阵或者更高维度的张量。
%
% 该函数会返回一个形式与 x 相同的值 y（标量、向量、矩阵或张量）。
function y = Sigmoid(x)
  y = 1 ./ (1 + exp(-x));
end
