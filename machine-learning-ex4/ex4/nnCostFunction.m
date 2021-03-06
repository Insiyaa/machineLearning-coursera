function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
k = num_labels;         
% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m, 1) X];         % 5000 x 401

% Modifying y in the required format
Y = zeros(m, k);        % 5000 x 10
for i=1:m
  num = y(i);
  Y(i, num) = 1;
endfor

z2 = Theta1 * X';           % 25 x 5000
A2 = 1 ./ (1 + e .^ -(z2)); % 25 x 5000
A2 = [ones(m, 1) A2'];      % 5000 x 26
z3 = Theta2 * A2';          % 10 x 5000
A3 = 1 ./ (1 + e .^ -(z3)); % 10 x 5000
 
% Training example wise summation as in the formula
for i = 1:m
  h = A3(:,i);
  J += (-(log(h)' * Y(i,:)') - (log(1 - h)' * (1 - Y(i,:)'))) ./ m; 
endfor

% Regularization
reg = 0;

% Row wise addition of square of elements.
for i = 1:hidden_layer_size
  reg += sum((Theta1(i, 2:end) .^ 2));
endfor

for i = 1:k
  reg += sum((Theta2(i, 2:end) .^ 2));
endfor

reg = (reg * lambda) / (2 * m);
J += reg;  

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m
  A1 = X(t, :);               % 1 x 401
  z2 = A1 * Theta1';          % 1 x 25
  A2 = sigmoid(z2);           % 1 x 25
  A2 = [1 A2];                % 1 x 26
  z3 = A2 * Theta2';          % 1 x 10
  A3 = sigmoid(z3);           % 1 x 10
  h = A3';                    % 10 x 1
  
  y_vect = Y(t,:);            % 1 x 10
  delta3 = h - y_vect';       % 10 x 1
  
  delta2 = (Theta2' * delta3) .* [1;sigmoidGradient(z2')]; % 26 x 1
  
  Theta1_grad += delta2(2:end) * A1; % 25 x 401
  Theta2_grad += delta3 * A2;        % 10 x 26
  
endfor

Theta1_grad /= m;
Theta2_grad /= m; 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
temp1 = Theta1_grad(:,1);
temp2 = Theta2_grad(:,1);
Theta1_grad += Theta1 .* (lambda / m);
Theta2_grad += Theta2 .* (lambda / m);

Theta1_grad(:,1) = temp1;
Theta2_grad(:,1) = temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
