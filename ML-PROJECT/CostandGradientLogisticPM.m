function [J, grad] = CostandGradientLogisticPM(X1,y,theta,lambda)


m = size(X1,1); % number of training examples


J = 0;
grad = zeros(size(theta));

J=(1/m)*(-(y'*log(sigmoid(X1*theta)))-((1-y)'*log(1-(sigmoid(X1*theta)))))+((lambda/(2*m))*(sum(theta(2:end).^2)));
grad(1)=(1/m)*(X1(:,1)'*(sigmoid(X1*theta)-y));
for i=2:size(theta,1),
  grad(i)=(1/m)*(X1(:,i)'*(sigmoid(X1*theta)-y))+((lambda/m)*theta(i));


end
%disp([(X1*theta) sigmoid(X1*theta) log(sigmoid(X1*theta))])

end
