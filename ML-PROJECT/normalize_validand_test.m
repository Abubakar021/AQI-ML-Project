function xnorm = normalize_validand_test(xval,mu,sigma)
%normalize validation and  test data
xnorm=zeros(size(xval));
m=size(xval,2);

for i=1:m,
  xnorm(:,i)=(xval(:,i)-mu(i));
  xnorm(:,i)=(xnorm(:,i))/sigma(i);

end

end

