function g = sigmoidGradient(z)

g = zeros(size(z));
for i=1:size(g,1),
  for j=1:size(g,2),
    g(i,j)=sigmoid(z(i,j))*(1-sigmoid(z(i,j)));
  endfor

end
