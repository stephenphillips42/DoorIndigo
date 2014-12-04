%%% SVM Regression
clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

Y = price_train;

% Use both training and testing data to make PCA
% nPCs = 500;
% [U,S,V] = svds([word_train bigram_train; word_test bigram_test],nPCs);
% Z = U*S;
load('pcaV500.mat')
Z = [word_train bigram_train]*VV;

% Train on half the data so we can cross-validate it
[trainind, testind] = crossvalind('HoldOut', length(Y), 0.5);

% Training
labels = zeros(size(Y));
labels(Y < mean(Y)) = 1;
labels(labels ~= 1) = 2;

labels_train = labels(trainind);
labels_test = labels(testind);
X = Z(trainind,:);
Xtest = Z(testind,:);
disp('Begin training...')
tic
trainSvm = svmtrain(X,labels_train,'kernel_function','rbf','rbf_sigma',12);
toc
labelsTrainErr = svmclassify(trainSvm,X);
labelsHat = svmclassify(trainSvm,Xtest);

fprintf('Training error by small SVM: %f%%\n',100*sum(labelsTrainErr ~= labels_train)/length(labels_train));
fprintf('Classification error by small SVM: %f%%\n',100*sum(labelsHat ~= labels_test)/length(labels_test));
%%
% LASSO
tic
[w, Fitinfo] = lasso(X(labelsTrain),Y,'Lambda',0.001);
toc


% For 
