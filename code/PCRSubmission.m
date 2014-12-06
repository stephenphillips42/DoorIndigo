%%% Principal Component Regression
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
nPCs = 1000;
[U,S,V] = svds([word_train bigram_train; word_test bigram_test],nPCs);
Z = U*S;

% LASSO
tic
[w, Fitinfo] = lasso(Z(1:size(word_test,1),:),Y,'Lambda',0.001);
toc

b = Fitinfo.Intercept(1);

Ztest = [word_test bigram_test] * V;

Yhat = Ztest*w + b;


