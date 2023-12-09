function ymod = modifylabels(y,no_labels)


m=size(y,1);
f=zeros(m,1);

%modifying the output labels
for i=1:m,
  if (y(i,:)>150),
    f(i,:)=1;
  %elseif (y(i)>50 && y(i)<=100),
   % f(i)=2;
  %elseif (y(i)>100 && y(i)<=150),
   % f(i)=3;
  %elseif (y(i)>150 && y(i)<=200),
   % f(i)=4;
  %elseif (y(i)>200 && y(i)<=300),
   % f(i)=5;
  %else,
   % f(i)=6;
  endif
end

ymod=f;
end
