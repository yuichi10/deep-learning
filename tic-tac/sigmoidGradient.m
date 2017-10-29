function g = sigmoidGradient(z)
% シグモイドの微分
g = zeros(size(z));

g = sigmoid(z) .* (1 - sigmoid(z));

end
