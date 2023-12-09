function r = relu(node)
  
k=size(node,1);

for i=1:k,
  if (node(i)<0),
    node(i)=0;
  endif
end
r=node;
end