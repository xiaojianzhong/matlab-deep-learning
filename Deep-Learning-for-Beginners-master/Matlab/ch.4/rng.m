% rng 函数设置随机数发生器的种子。
%
% x 是一个标量。
function rng(x)
  randn('seed', x)
  rand('seed', x)
end
