clear;

%Testing convenience (always make sure I'm in the right directory
%cd('/mnt/castor/seas_home/p/pballen/ML/DoorIndigo/code')

load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat



Xall = [city_train word_train];
Yall = price_train;

[rest, intrain] = crossvalind('LeaveMOut', size(Yall, 1), 7000);

Xtrain = Xall(intrain, :);
Ytrain = Yall(intrain, :);
Xtest = Xall(rest, :);
Ytest = Yall(rest, :);

[whocares, intest] = crossvalind('LeaveMOut', size(Ytest, 1), 5000);
Xtest = Xtest(intest, :);
Ytest = Ytest(intest, :);


clear Xall Yall price_train whocares
clear bigram_test
clear bigram_train
clear city_test
clear city_train
clear word_test
clear word_train


lambda = .1;		% regularization constant
kernel = 'gauss';	% kernel type
sigma = 15;			% Gaussian kernel width


[alpha,Ytest_est] = km_krr(Xtrain,Ytrain,kernel,sigma,lambda,Xtest);	% regression weights alpha, and output y2 of the regression test


norm(Ytest - Ytest_est) / sqrt(size(Ytest, 1))


