function [theta] = training_model_LinearPM(X, y, lambda)


% Initialize Theta
initial_theta = zeros(size(X, 2), 1);


costFunction = @(t) CostandGradientLinearPM(X, y, t, lambda);


options = optimset('MaxIter', 200, 'GradObj', 'on');


theta = fminunc(costFunction, initial_theta, options);

end
