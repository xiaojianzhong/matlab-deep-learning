% 该文件展示了在正向传播的过程中图像是如何一步步被处理的。

clear all

load('MnistConv.mat') % 加载预训练模型

k  = 2; % 取训练集中第 2 张图片作为例子进行说明
x  = X(:, :, k);
y1 = Conv(x, W1);
y2 = ReLU(y1);
y3 = Pool(y2);
y4 = reshape(y3, [], 1);
v5 = W5*y4;
y5 = ReLU(v5);
v  = Wo*y5;
y  = Softmax(v);


figure;
display_network(x(:)); % 显示输入的图片
title('Input Image')

convFilters = zeros(9*9, 20);
for i = 1:20
  filter            = W1(:, :, i);
  convFilters(:, i) = filter(:);
end
figure
display_network(convFilters); % 显示卷积层用到的过滤器
title('Convolution Filters')

fList = zeros(20*20, 20);
for i = 1:20
  feature     = y1(:, :, i);
  fList(:, i) = feature(:); % 显示图片经过卷积处理后的结果
end
figure
display_network(fList);
title('Features [Convolution]')

fList = zeros(20*20, 20);
for i = 1:20
  feature     = y2(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList); % 显示图片经过卷积 + ReLU 处理后的结果
title('Features [Convolution + ReLU]')

fList = zeros(10*10, 20);
for i = 1:20
  feature     = y3(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList); % 显示图片经过卷积 + ReLU + 池化处理后的结果
title('Features [Convolution + ReLU + MeanPool]')
