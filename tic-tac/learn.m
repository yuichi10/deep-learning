clear; close all; clc

input_layout_size = 9;

% 9ます + pass
output_layout_size = 10;

fprintf('try by small number of training\n')
load('smallx.txt');
smallx = smallx';
load('smally.txt');
smally = smally';
[J, grad1, grad2] = costFunction(rand(15, 10) + 2, rand(9, 16) + 1, smallx, smally, 0);
J;
grad1;
grad2;

fprintf('棋譜データの読みこみ...\n')

load('X.txt');
X = X';
load('Y.txt');
Y = Y';
theta1 = rand(15, 10) + 1;
theta2 = rand(9, 16) + 1;

fprintf('calculate theta')
for i = 1:1500
	[J, grad1, grad2] = costFunction(theta1 - grad1, theta2 - grad2, X, Y, 1);
    J 
    theta1 = theta1 - grad1;
	theta2 = theta2 - grad2;
end
save theta.txt theta1 theta2;

