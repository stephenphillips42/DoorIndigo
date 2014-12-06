clear;

%Testing convenience (always make sure I'm in the right directory
%cd('/mnt/castor/seas_home/p/pballen/ML/DoorIndigo/code')
tic
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

Xall = [city_train word_train];
Yall = price_train;

% Principal components
load('pcaV500','VV');
Zall = [word_train bigram_train] * VV;

[rest, intrain] = crossvalind('LeaveMOut', size(Yall, 1), 7000);

% Using Kmeans and PCA (maybe too much approximation)
K = 2000;
load('kmeansStuff2000.mat')
clusterMeans = zeros(K,size(Zall,2));
clusterPrices = zeros(K,1);
for i = 1:K
    clusterMeans(i,:) = mean(Zall(clusterIds==i,:));
    clusterPrices(i) = mean(Yall(clusterIds==i));
end

Xtrain = Xall(intrain, :);
% Ztrain = Zall(intrain, :);
Ztrain = clusterMeans;
% Ytrain = Yall(intrain, :);
Ytrain = clusterPrices;
Xtest = Xall(rest, :);
Ztest = Zall(rest, :);
Ytest = Yall(rest, :);

% [~, intest] = crossvalind('LeaveMOut', size(Ytest, 1), 5000);
% Xtest = Xtest(intest, :);
% Ztest = Ztest(intest, :);
% Ytest = Ytest(intest, :);
toc

clear Xall Yall Zall price_train
clear bigram_test
clear bigram_train
clear city_test
clear city_train
clear word_test
clear word_train


% lambda = .00005;		% regularization constant
kernel = 'gauss';	% kernel type
% sigma = 35;			% Gaussian kernel width

Sigmas = 20:50;
Lambdas = logspace(1,-6,30);
err = zeros(length(Sigmas),length(Lambdas));
s = 1;
l = 1;
time1 = tic;
for s = 1:length(Sigmas)
    sigma = Sigmas(s);
    for l = 1:length(Lambdas)
        lambda = Lambdas(l);
        tic
        [alpha,Ytest_est] = km_krr(Ztrain,Ytrain,kernel,sigma,lambda,Ztest);	% regression weights alpha, and output y2 of the regression test
        toc
        err(s,l) = norm(Ytest - Ytest_est) / sqrt(size(Ytest, 1));
    end
end
toc(time1)

imagesc(err);
xlabel('Lambdas')
ylabel('Sigmas')
colormap;
a1 = gca;
a1.XTickLabel = cellfun(@(x) sprintf('%f',x), num2cell(Lambdas)); 







