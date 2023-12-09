function [X_norm, mu, sigma] = featureNormalization(X)


%mu = mean(X);
%X_norm = bsxfun(@minus, X, mu);
X_scal=zeros(size(X));
mu=zeros(1,size(X,2));
sigma=zeros(1,size(X,2));
X_norm=zeros(size(X));
%sigma = std(X_norm);
m=size(X,2);
for i=1:m,
  mu(i)=mean(X(:,i));
  X_scal(:,i)=X(:,i)-mu(i);
  sigma(i) = std(X_scal(:,i));
  X_norm(:,i)=X_scal(:,i)/sigma(i);

%X_norm = bsxfun(@rdivide, X_norm, sigma);


end

end
