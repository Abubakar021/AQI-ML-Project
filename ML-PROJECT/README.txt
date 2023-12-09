Download the folder and run main.m using Octave's command prompt. The model first trains itself using the provided training data set( it will take several minutes to train) and then will find the suitable lambda(regularization parameter) through learning curves by using cross-validation data set (this will take more time than the time taken for training). Finally after obtaining suitable parameters (weights), the trained model was tested by a provided test data set.
 
1) For linear regression ML model run mainLinearPM.m file
2) For logistic regression ML model run mainLogisticPM.m file
3) For artificial neural network ML model run mainANN.m file

Few changes need to made in the above files whenever we load the data, so whether to build ML model using single feature or using multiple features. (Default is set to import/load multiple features)