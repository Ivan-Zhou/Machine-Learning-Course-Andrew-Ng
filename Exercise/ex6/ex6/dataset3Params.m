function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

list = [0.01 0.03 0.1 0.3 1 3 10 30]; % List of 8 values to loop through
list_length = length(list); % Length of the list
pred_err = zeros(list_length^2,1); % Vector to store the prediction errors
Cs = zeros(list_length^2,1); % Vector to store all the C
sigmas = zeros(list_length^2,1); % Vector to store all the sigmas
count = 0;

for C = list
    for sigma = list
        count = count+1;
        % Train the model
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        % Make predictions
        predictions = svmPredict(model,Xval);
        % Compute and store the prediction error
        pred_err(count) = mean(double(predictions ~= yval));
        Cs(count) = C;
        sigmas(count) = sigma;
    end
end

% Find the minimum prediction error and corresponding C & sigma
[~,pred_err_min] = min(pred_err);
C_min = Cs(pred_err_min); % Find the corresponding C
sigma_min = sigmas(pred_err_min); % Find the corresponding sigma

% Prepare to return the value
C = C_min;
sigma = sigma_min;

% =========================================================================

end
