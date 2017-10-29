clear; close all; clc

input_layout_size = 9;

% 9ます + pass
output_layout_size = 10;

fprintf('棋譜データの読みこみ...\n')

load('X.txt');
X = X';
load('Y.txt');
Y = Y';
[J, grad1, grad2] = costFunction(zeros(15, 10) + 1, zeros(9, 16) + 1, X, Y, 0);
J
size(grad1)
size(grad2)
