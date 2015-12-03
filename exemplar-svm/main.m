clear all 
%Define folder where database is located
folder = 'C:\Users\eleves\Documents\Joao_Matlab\lfw';

%Define person to identify
person = 'George_W_Bush'; 

%Extract filenames and run the descriptor on them
images = Create_NameFile_Vector(folder);
features_cell = descriptor(images);
features = vectorizer(features_cell);

%Create positive labels for person of interest and negative for the rest
labels = label_identify(person, images);

lambda = 0.01 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations


save('dataset.mat', 'features', 'labels');

%Create train and data sets by shuffling the features vector and splitting
split_position = 10000;
[train_set, test_set]=split([features; labels'], split_position);
%load('dataset.mat', 'features', 'labels');
%Create an SVM based on the features and labels
[w b info] = vl_svmtrain(train_set(1:end-1,:), train_set(end,:)', lambda, 'MaxNumIterations', maxIter);