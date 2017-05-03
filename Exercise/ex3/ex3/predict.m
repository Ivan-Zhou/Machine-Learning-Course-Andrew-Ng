function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% First Hidden Layer
% Add ones to the X data matrix
X = [ones(m, 1) X]; % Bias Column
% Generate the Hidden Layer
% Dimension of Theta1 = 25*401
A = sigmoid(X*Theta1');

% Output Layer
% Add ones to the A data matrix
A = [ones(m, 1) A]; % Bias Column
% Generate the Output
% Dimension of Theta2 = 10*26
result = sigmoid(A*Theta2');

[~,p]=max(result,[],2); % Find the Max in each Row: The column number equals the label 









% =========================================================================


end
