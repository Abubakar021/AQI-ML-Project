function [error_train, error_val] = ...
    LearningcurveLogisticPM(X, y, Xval, yval, lambda)


m = size(X, 1);


error_train = zeros(m, 1);
error_val   = zeros(m, 1);



for i=1:m,
  ti=training_model_LogisticPM(X(1:i,:),y(1:i),lambda);
  [Jt gra]=CostandGradientLogisticPM(X(1:i,:), y(1:i), ti, 0);
  [Jv gre]=CostandGradientLogisticPM(Xval, yval, ti, 0);
  error_train(i)=Jt;
  error_val(i)=Jv;
endfor


end
