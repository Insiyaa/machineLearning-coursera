function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
cost = computeCost(X, y, theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    y1 = X * theta;
    der = (sum(y1 - y))/m;
    temp1 = theta(1) - (alpha * der);
    der = (sum((y1 - y)' * (X(:,2))))/m;
    temp2 = theta(2) - (alpha * der);
    theta(1) = temp1;
    theta(2) = temp2;
    costNew = computeCost(X, y, theta);
    if costNew < cost
      cost = costNew;
    else
      break
    end;





end;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end


