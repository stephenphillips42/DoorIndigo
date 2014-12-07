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
nPCs = 500;
[U,S,V] = svds([word_train bigram_train; word_test bigram_test],nPCs);
Z = U*S;

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

fprintf('Training error by SVM: %f%%\n',100*sum(labelsTrainErr ~= labels_train)/length(labels_train));
fprintf('Classification error by SVM: %f%%\n',100*sum(labelsHat ~= labels_test)/length(labels_test));

% LASSO
tic
[w1, Fitinfo1] = lasso(X(labelsTrain == 1),Y,'Lambda',0.001);
[w2, Fitinfo2] = lasso(X(labelsTrain == 2),Y,'Lambda',0.001);
toc

X_test = Z(length(Y)+1:end,:);
Yhat = zeros(size(X_test,1),1);

labels_test = svmclassify(trainSvm,X_test);
Yhat(labels_test == 1) = X_test(labels_test == 1)*w1 + FitInfo1.Intercept(1);
Yhat(labels_test == 2) = X_test(labels_test == 2)*w2 + FitInfo2.Intercept(1);

% Test Yhat error here


