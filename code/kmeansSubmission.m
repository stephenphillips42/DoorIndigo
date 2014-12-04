%%% Kmeans
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

% Run kmeans using MATLAB's built in
K = 200; % Number of clusters
tic
clusterIds = kmeans(Z,K);
toc

%
clusterMeans = zeros(K,size(Z,2));
clusterPrices = zeros(K,1);
for i = 1:K
    clusterMeans(i,:) = mean(Z(clusterIds==i,:));
    clusterPrices(i) = mean(Y_train(clusterIds==i));
end

% Radial basis functions with the means
sigma = 8;
rbf_train = zeros(length(Y_train),K);
for i = 1:K
    rbf_train(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Z),1)-Z).^2,2)/(2*sigma^2));
end

% Create training and testing sets
X = [city_train Z rbf_train];

% LASSO
tic
[w, Fitinfo] = lasso(X,Y,'Lambda',0.001);
toc
b = Fitinfo.Intercept(1);

Ztest = [word_test bigram_test] * V;

rbf_test = zeros(size(Ztest,1),K);
for i = 1:K
    rbf_test(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Ztest),1)-Ztest).^2,2)/(2*sigma^2));
end

Xtest = [city_test Ztest rbf_test];

Yhat = Xtest*w + b;


