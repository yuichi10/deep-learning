function [result] = predict(x)
load('theta.txt');

size(x)
a1 = [1; x];
z2 = theta1 * a1;
a2 = sigmoid(z2);
a2 = [1; a2];
z2 = theta2 * a2;
a3 = sigmoid(z2);
a3
 
