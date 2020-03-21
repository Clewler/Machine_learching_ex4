function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
a1 = [ones(m,1) X];
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

reg_param = (lambda/(2*m)) * sum(sum(theta(2:end).^2));

for i=1:m,
   tmp = X(i,:) * theta;
   J = J + (tmp - y(i))^2;
   
   grad(1) = grad(1) + (tmp - y(i))*X(i,1);
   for j=2:length(X(i,:)),
         grad(j) = grad(j) + ((tmp - y(i))*X(i,j)) + ((lambda/m) * theta(j));
   end
end

J = (1/(2*m)) * J + reg_param;
grad = (1/m) * grad;


% =========================================================================

grad = grad(:);

end
