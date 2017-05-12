function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_t = X*theta; % Hypothesis h theta, whose dimension becomes m*1
J = 1/(2*m)*sum((h_t-y).^2) + lambda/(2*m)*sum(theta(2:end).^2);; % Cost
grad(1) = 1/m*sum(X(:,1)'*(h_t-y),2); % The gradient of theta 0
grad(2:end) = 1/m*sum(X(:,2:end)'*(h_t-y),2) + lambda/m.*theta(2:end); % The gradient of the remaining thetas





% =========================================================================

end
