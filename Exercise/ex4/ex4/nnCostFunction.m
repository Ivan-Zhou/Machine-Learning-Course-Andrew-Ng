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
% nn_params: Weights Theta Vector
% input_layer_size: column size of the input layer (number of pixels), single digit
% hidden_layer_size: number of units in the hidden layer, single digit
% num_labels: size of the output layer


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Compute the nodes on the second layer
X_input = [ones(m,1) X]; % Add a bias layer to X; dimension: m * (input_layer_size+1)
z2 = X_input*Theta1';
a2 = sigmoid(z2); % dimension: m * hidden_layer_size +1;

% Compute the nodes on the output layer
a2_input = [ones(m,1) a2]; % Add a bias layer to a2; dimension: m * (hidden_layer_size+1);
z3 = a2_input*Theta2';
ht = sigmoid(z3);
J = 0; % Initialize the J
for i = 1:num_labels % Loop through each label
    y_label = y == i;
    ht_label = ht(:,i);
    J = J + 1/m*sum(-y_label.*log(ht_label)-(1-y_label).*log(1-ht_label));
end
% We cannot add the regularization in the loop; otherwise it will be added
% repeatedly
J =  J + lambda/(2*m)*(sum(sum(Theta2(:,2:end).^2))+sum(sum(Theta1(:,2:end).^2)));

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
% Partial Derivative of the cost function with respect to Theta1

delta3 = zeros(m,num_labels);
delta2 = zeros(m, hidden_layer_size);

for t = 1:m
    for i = 1:num_labels
        y_label = y(t)==i; % Logical variable
        delta3(t,i) = ht(t,i) - y_label;
    end
    % Hidden layer l = 2
    % Theta2: [num_labels,(hidden_layer_size + 1)]
    % delta3: [m, num_labels]
    % z2: [m, (hidden_layer_size + 1)];
    % As a result: delta2: [m, (hidden_laber_size)]
    %size(delta3(t,:))
    delta2(t,:) = delta3(t,:)*Theta2(:,2:end).*sigmoidGradient(z2(t,:));
    
    
end

% delta3: [m, num_labels]
% a2: [m, hidden_layer_size]
%Delta2 = Delta2 + delta3 * a2'

% delta3: [m, num_labels]
% a2_input: m * (hidden_layer_size +1)
% Delta2: num_labels, (hidden_layer_size + 1)
Delta2 = delta3' * a2_input;
% Theta2_grad: num_labels, (hidden_layer_size + 1)
Theta2_grad = 1/m .* Delta2;

% delta2: m * hidden_laber_size
% X_input: m * (input_layer_size + 1)
% Delta1: hidden_layer_size, (input_layer_size + 1)
Delta1 = delta2' * X_input;
% Theta1_grad = hidden_layer_size, (input_layer_size + 1)
Theta1_grad = 1/m .* Delta1;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+lambda/m.*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+lambda/m.*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
