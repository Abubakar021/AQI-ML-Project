function [theta] = training_model_LogisticPM(X, y, lambda)


% Initialize Theta
initial_theta = zeros(size(X, 2), 1);


costFunction = @(t) CostandGradientLogisticPM(X, y, t, lambda);


options = optimset('MaxIter', 200, 'GradObj', 'on');


theta = fminunc(costFunction, initial_theta, options);

end
