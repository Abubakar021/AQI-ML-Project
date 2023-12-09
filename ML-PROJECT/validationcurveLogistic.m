function [lambda_vec, error_train, error_val] = ...
    validationcurveLogistic(X, y, Xval, yval)


% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 15 20 25 30]';


error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i=1:length(lambda_vec),
  ti=training_model_LogisticPM(X,y,lambda_vec(i));
  [Jt gra]=CostandGradientLogisticPM(X, y, ti, 0);
  [Jv gre]=CostandGradientLogisticPM(Xval, yval, ti, 0);
  error_train(i)=Jt;
  error_val(i)=Jv;


endfor







% =========================================================================

end
