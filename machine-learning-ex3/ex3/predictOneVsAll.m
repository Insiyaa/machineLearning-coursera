function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% WHAT I UNDERSTAND - Insiyah
% First we calculate a matrix having theta0x0 + theta1x1 + .. and so on. And then
% the sigmoid of that function. That would be a 10x5000 matrix and we call it A
% standing for activation values. These are probability of each class.
% The max returns a row vector of max values of each column, i.e max_val. The class 
% vector would give that from which class the max value belonged to and hence that 
% would be the predicted class of the image. It's a row vector, hence we return
% its transpose
 
ginv_a = all_theta * X';
A = 1 ./ (1 + e .^ -(ginv_a));
[max_val, class] = max(A);

% fprintf("Sum of probability of first column is %f", sum(A(:,1)));
% This is nearly equal to 1. Hence sum of activation values is equal to 1.

p = class';

% =========================================================================


end
