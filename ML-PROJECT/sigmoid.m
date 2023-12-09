function g = sigmoid(z)
%computes the sigmoid of z.
g = zeros(size(z));
g = 1./(1+exp(-z));
%for i=1:size(z,1),
%  for j=1:size(z,2),
%    g(i,j)=1/(1+exp(-z(i,j)));
 % endfor

%end

end
