clear;
close all;
clc;

%load the training set data
train_data=load("more_features.txt");
shuffle_data_train=train_data(randperm(size(train_data,1)),:);
x=shuffle_data_train(:,1:12);%input data

y=shuffle_data_train(:,13);%output data

fprintf('Visualizing Data\n')
figure(1);
plot_data(x,y);

%hyper-parameters
input_layer_size=12;
hidden_layer1_size =30;
hidden_layer2_size = 30;
no_labels=1;
lambda=0;
epochs=50;

%optimizing the neural network by performing feature normalization
[x_norm, mu, sigma]=featureNormalization(x);
x=x_norm;

%randomly initializing weights using initializeweights function
fprintf("randomly initializing weights\n")
theta1= InitializeWeights(input_layer_size,hidden_layer1_size);
theta2= InitializeWeights(hidden_layer1_size,hidden_layer2_size);
theta3= InitializeWeights(hidden_layer2_size,no_labels);
%disp(size(theta1));
%disp(size(theta2));
%disp(size(theta3));
%disp(theta1);
%disp(theta2);
initialweights=[theta1(:);theta2(:);theta3(:)];%unrolling the parameters

%training the neural network
%modifiedweights=size(initialweights);
fprintf('\nTraining Neural Network with all features without regularization...\n')

[modifiedweights, cost] = training_model(initialweights, ...
                                         input_layer_size, ...
                                         hidden_layer1_size, ...
                                         hidden_layer2_size,no_labels, ...
                                         x, y, lambda,epochs);

% Obtain Theta1 and Theta2 back from modifiedweights
Theta1_mod = reshape(modifiedweights(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2_mod= reshape(modifiedweights(1+(hidden_layer1_size * (input_layer_size + 1)):...
                 (hidden_layer2_size * (hidden_layer1_size+ 1)+hidden_layer1_size * (input_layer_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3_mod = reshape(modifiedweights((1+(hidden_layer1_size * (input_layer_size + 1))+hidden_layer2_size * (hidden_layer1_size+ 1)):end), ...
                 no_labels, (hidden_layer2_size + 1));


%disp(size(Theta1_mod));
%disp(size(Theta2_mod));
%disp(size(Theta3_mod));
%examining training set accuracy
y_mod=modifylabels(y,no_labels);
%y_mod=Modify_Labels_num(y_mod);
%[max_num yvec]=max(y_mod, [], 2);


pred_val_train = predictedlabels(Theta1_mod, Theta2_mod,Theta3_mod, x);
%pred_val_train=Modify_Labels_num(pred_val_train);
%[max_num1 pred_trainvec]=max(pred_val_train, [], 2);
%disp([y_mod pred_val_train]);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_val_train == y_mod)) * 100);


%%Now we find the value of lambda which gives the lowest error on cross-validation data set
%%Loading the cross-validation data-set

validation_data=load("more_feature_valid.txt");
shuffle_data_valid=validation_data(randperm(size(validation_data,1)),:);
xval=shuffle_data_valid(:,1:12);
yval=shuffle_data_valid(:,13);

%normalizing the cross-validation data
xval_norm = normalize_validand_test(xval,mu,sigma);
%xval_scal=bsxfun(@minus,xval, mu);
%xval_norm=bsxfun(@rdivide, xval_scal, sigma);
xval=xval_norm;

%evaluating the validation data set accuracy
yval_mod=modifylabels(yval,no_labels);
%yval_mod=Modify_Labels_num(yval_mod);
%[max_num yval_vec]=max(yval_mod, [], 2);
%yval_vec

pred_val_valid = predictedlabels(Theta1_mod, Theta2_mod,Theta3_mod, xval);
%pred_val_valid = Modify_Labels_num(pred_val_valid);
%[max_num1 pred_validvec]=max(pred_val_valid, [], 2);
%pred_validvec

fprintf('validation Set Accuracy: %f\n', mean(double(pred_val_valid == yval_mod)) * 100);

%evaluating whether the model has high bias/variance by plotting learning curve.
fprintf("\nPlotting learning curves\n");
m=size(xval,1);
[train_error, valid_error] =learningCurve(initialweights,input_layer_size,...
                                     hidden_layer1_size,hidden_layer2_size,no_labels,x, y, xval, yval, lambda,epochs);
figure(5);
plot(1:m,train_error, 1:m, valid_error);
title('Learning curve of the model')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')


%we will to choose the value of lambda(regularization-parameter) which
%minimizes the error on cross-validation data set
fprintf("\nFinding the optimal value of regularization parameter\n");
[lambda_values, training_error, validation_error]=optimlambdaCurve(initialweights,input_layer_size,...
                                                                   hidden_layer1_size,
                                                                   hidden_layer2_size,no_labels,...
                                                                   x, y, xval, yval,epochs);
figure(6);
plot(lambda_values,training_error,lambda_values,validation_error);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
title("Optimal regularization parameter Lambda");
%Training the model(lambda=0.3)
lambda=0.3;
weights_reg=size(initialweights);
fprintf('\nTraining Neural Network with regularization...\n')

[weights_reg, cost] = training_model(initialweights, ...
                                         input_layer_size, ...
                                         hidden_layer1_size, ...
                                         hidden_layer2_size,no_labels, ...
                                         x, y, lambda,epochs);
%weights_reg=modifiedweights;
% Obtain Theta1 and Theta2 back from modifiedweights
Theta1_reg = reshape(weights_reg(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2_reg= reshape(weights_reg(1+(hidden_layer1_size * (input_layer_size + 1)):...
                 (hidden_layer2_size * (hidden_layer1_size+ 1)+hidden_layer1_size * (input_layer_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3_reg = reshape(weights_reg((1+(hidden_layer1_size * (input_layer_size + 1))+hidden_layer2_size * (hidden_layer1_size+ 1)):end), ...
                 no_labels, (hidden_layer2_size + 1));

%examining training set accuracy
pred_val_trainreg = predictedlabels(Theta1_reg, Theta2_reg,Theta3_reg, x);
%pred_val_trainreg=Modify_Labels_num(pred_val_trainreg);
%[max_num1 pred_trainvec]=max(pred_val_train, [], 2);
%disp([y_mod pred_val_train]);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_val_trainreg == y_mod)) * 100);

%examining validation set accuracy
pred_val_validreg = predictedlabels(Theta1_reg, Theta2_reg,Theta3_reg, xval);
%pred_val_validreg = Modify_Labels_num(pred_val_validreg);
%[max_num1 pred_validvec]=max(pred_val_valid, [], 2);
%pred_validvec

fprintf('validation Set Accuracy: %f\n', mean(double(pred_val_validreg == yval_mod)) * 100);

% calculating the test-data set accuracy
test_data=load("more_feature_test.txt");
shuffle_data_test=test_data(randperm(size(test_data,1)),:);
xtest=shuffle_data_test(:,1:12);
ytest=shuffle_data_test(:,13);

xtest_norm = normalize_validand_test(xtest,mu,sigma);
xtest=xtest_norm;
ytest_mod=modifylabels(ytest,no_labels);
%ytest_mod=Modify_Labels_num(ytest_mod);
%fprintf("\nComparing predicted outputs and actual outputs\n");
%fprintf("Predicted values   Actual values\n");

%examining test set accuracy
pred_val_testreg = predictedlabels(Theta1_reg, Theta2_reg,Theta3_reg, xtest);
%pred_val_testreg = Modify_Labels_num(pred_val_testreg);
%display([pred_val_testreg,ytest_mod]);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_val_testreg == ytest_mod)) * 100);


