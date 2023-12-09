function [error_train, error_val] = ...
    LearningcurvePM(X, y, Xval, yval, lambda)



m = size(X, 1);


error_train = zeros(m, 1);
error_val   = zeros(m, 1);



for i=1:m,
  ti=training_model_LinearPM(X(1:i,:),y(1:i),lambda);
  [Jt gra]=CostandGradientLinearPM(X(1:i,:), y(1:i), ti, 0);
  [Jv gre]=CostandGradientLinearPM(Xval, yval, ti, 0);
  error_train(i)=Jt;
  error_val(i)=Jv;
endfor


end
