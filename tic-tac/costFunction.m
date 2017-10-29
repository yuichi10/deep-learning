function [J, theta1_grad, theta2_grad] = costFunction(theta1, theta2, x, y, lambda)
% コストファンクションの実装
% コストと次のthetaの計算に使うgradをかえす
% X 9 * 500 => biasとして1を追加するので 10 * 500 またきちんと並べ替えて 10 * 500でやる
% y 9 * 500
% theta1 15 * 9 => biasの分 15 * 10
% theta2 9 * 15 => biasの分 9 * 16
J = 0;
% TODO: 間違ってる　修正が必要
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

m = size(x, 2);

a1 = [x; zeros(1, size(x, 2)) + 1];
z2 = theta1 * a1;
a2 = sigmoid(z2);
a2 = [a2; zeros(1, size(a2, 2)) + 1];
z2 = theta2 * a2;
a3 = sigmoid(z2);

% calculte cost function
size(a3)
size(y)
a3(:, 4)
% y .* log(a3) + (1-y)
J = sum(sum(y .* log(a3) + (1 - y) .* log(1-a3))) / m * -1;
J
J = J + lambda / (2 * m) * (sum(sum(theta1 .* theta1)) + sum(sum(theta2 .* theta2)));
x = [x; zeros(1, size(x, 2)) + 1];

for i = 1:m
  % set a(i) = x(i)
  % perform foward propagation to compute
  a1 = x(:, i);
  z2 = theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [a2; zeros(1, size(a2, 2)) + 1];
  z3 = theta2 * a2;
  a3 = sigmoid(z3);
  % using y(i), compute delta(L)
  delta3 = (a3 - y(:, i));

  % z2 にバイアスを追加(theta2にはバイアスが含まれているため)
  z2 = [1; z2];
  % compute delta(L-1), delta(L-2)...
  delta2 = (theta2' * delta3) .* sigmoidGradient(z2);
  % delta2のバイアスを消す
  delta2 = delta2(2:end);
  
  theta1_grad = theta1_grad + delta2 * a1';
  theta2_grad = theta2_grad + delta3 * a2';
end

theta1_grad(:, 1) = theta1_grad(:, 1) / m;
theta1_grad(:, 2:end) = theta1_grad(:, 2:end) + lambda * theta1(:, 2:end);
theta2_grad(:, 1) = theta2_grad(:, 1) / m;
theta2_grad(:, 2:end) = theta2_grad(:, 2:end) / m + lambda * theta2(:, 2:end);













