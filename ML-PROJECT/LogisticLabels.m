function Logisticoutput=LogisticLabels(Input_vector, threshold_value)

m=size(Input_vector,1);
Logisticoutput=zeros(m,1);
for i=1:m,
  if Input_vector(i)>=threshold_value,
    Logisticoutput(i)=1;
  endif
endfor

end

