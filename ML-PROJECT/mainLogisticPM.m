clear;
close all;
clc;
fprintf('...Logistic Regression Model with all features...\n')
%load the training set data
train_data=load("more_features.txt");
shuffle_data_train=train_data(randperm(size(train_data,1)),:);
x=shuffle_data_train(:,1:12);%input data
y=shuffle_data_train(:,13);%output data
m = length(y); % number of training examples

[x_norm, mu, sigma]=featureNormalization(x);
X=x_norm;

X = [ones(m, 1), X]; % Add a column of ones to x
ymod=modifylabels(y,1);
%disp([y ymod])

lambda=0;
threshold_value=0.5;

fprintf('Training the model without regularization\n')
theta = training_model_LogisticPM(X, ymod, lambda);


%disp(ymod)

LogisticPredictions=sigmoid(X*theta);
LogisticPredictions=LogisticLabels(LogisticPredictions,threshold_value);
%LinearPredictionsmod=modifylabels(LinearPredictions,1);
%disp(LinearPredictionsmod);
%disp([LinearPredictionsmod ymod])
fprintf('Training set Accuracy: %f\n', mean(double(LogisticPredictions == ymod)) * 100);

validation_data=load("more_feature_valid.txt");
shuffle_data_valid=validation_data(randperm(size(validation_data,1)),:);
xval=shuffle_data_valid(:,1:12);
yval=shuffle_data_valid(:,13);

xval_norm = normalize_validand_test(xval,mu,sigma);
xval=xval_norm;
xval= [ones(length(yval), 1), xval];
yval_mod=modifylabels(yval,1);

LogitP_valid=sigmoid(xval*theta);
LogitP_valid=LogisticLabels(LogitP_valid,threshold_value);
fprintf('\nvalidation Set Accuracy: %f\n', mean(double(LogitP_valid == yval_mod)) * 100);





%Analyzing Learning curves
fprintf('\nAnalyzing the model through Learning Curve\n');
[error_train, error_val] = ...
    LearningcurveLogisticPM(X, ymod, ...
                  xval, yval_mod,...
                  lambda);
disp([error_train error_val])
figure(1)
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')


%Choosing best lambda(regularization parameter) value
fprintf('\nFinding best value for regularization parameter\n');
[lambda_vec, error_train, error_val] = ...
           validationcurveLogistic(X, ymod, xval, yval_mod);

figure(2)
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');



fprintf('\nTraining the model with regularization parameter ...\n')
lambda=0;
theta = training_model_LogisticPM(X, ymod, lambda);

LogisticPredictions=sigmoid(X*theta);
LogisticPredictions=LogisticLabels(LogisticPredictions,threshold_value);
fprintf('Training set Accuracy: %f\n', mean(double(LogisticPredictions== ymod)) * 100);

LogitP_valid=sigmoid(xval*theta);
LogitP_valid=LogisticLabels(LogitP_valid,threshold_value);
fprintf('\nvalidation Set Accuracy: %f\n', mean(double(LogitP_valid == yval_mod)) * 100);


% calculating the test-data set accuracy
test_data=load("more_feature_test.txt");
shuffle_data_test=test_data(randperm(size(test_data,1)),:);
xtest=shuffle_data_test(:,1:12);
ytest=shuffle_data_test(:,13);


xtest_norm = normalize_validand_test(xtest,mu,sigma);
xtest=xtest_norm;
xtest=[ones(length(ytest), 1), xtest];
ytest_mod=modifylabels(ytest,6);

LogitP_test=sigmoid(xtest*theta);
LogitP_test=LogisticLabels(LogitP_test,threshold_value);
%disp([LP_testmod ytest_mod])
fprintf('\nTest Set Accuracy: %f\n', mean(double(LogitP_test== ytest_mod)) * 100);

