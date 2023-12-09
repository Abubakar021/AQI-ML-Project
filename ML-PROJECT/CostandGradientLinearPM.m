function [J, grad] = CostandGradientLinearPM(X, y, theta, lambda)



m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


J=(1/(2*m))*(sum(((X*theta)-y).^2))+(lambda/(2*m))*(sum((theta(2:end).^2)));

grad(1,:)=(1/m)*(sum(X(:,1)'*((X*theta)-y)));
for i=2:size(theta,1),
  grad(i,:)=(1/m)*(sum(X(:,i)'*((X*theta)-y)))+(lambda/m)*theta(i,:);

endfor










grad = grad(:);

end
