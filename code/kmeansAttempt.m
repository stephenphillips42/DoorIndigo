%% Kmeans
clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X_train =[city_train word_train bigram_train];
[N,p] = size(X_train);
Y_train = price_train;
X_test = [city_test word_test bigram_test];
Y_city = cell2mat(cellfun(@(x) find(x,1,'first'), num2cell(city_train,2),'UniformOutput',false));

load('pcaV.mat','V');
Z = [word_train bigram_train] * V;

K = 100; % Number of clusters
loading = true;

%% Rerun kmeans
if loading
    load('kmeansStuff100_small.mat')
else
    tic
    clusterIds = kmeans(Z,K);
    toc
end

%%
clusterMeans = zeros(K,size(Z,2));
clusterPrices = zeros(K,1);
for i = 1:K
    clusterMeans(i,:) = mean(Z(clusterIds==i,:));
    clusterPrices(i) = mean(Y_train(clusterIds==i));
end
save('kmeansStuff5.mat','Z','clusterMeans','clusterIds','clusterPrices');

%
% figure;
% plot3(clusterMeans(:,1),clusterMeans(:,2),clusterPrices,'r.');

%%
figure; hold on
plotpc1=1;
plotpc2=2;
plotpc3=3;
cc = hsv(K);
for i = 1:K
    fprintf('Cluster %d, with %d members, range %f\n',i,sum(clusterIds==i),range(Y_train(clusterIds==i)))
%     scatter3(Z(clusterIds==i,plotpc1),...
%           Z(clusterIds==i,plotpc2),...
%           Z(clusterIds==i,plotpc3),...
%           (Y_train(clusterIds==i)).^11/10000000000,...
%           100*(Y_train(clusterIds==i)),...
%           '.');
    plot3(Z(clusterIds==i,plotpc1),Z(clusterIds==i,plotpc2),Y_train(clusterIds==i),'.','color',cc(i,:));
    hold on
end
% colormap('gray')
colorbar
xlabel('Principal Components')
ylabel('Price')
hold off
pause(3)

%% Radial basis functions with the means
% Compute the feature vectors
sigma = 10; % TODO: Tune this. Question: HOW? Lasso takes WAY too long.
rbf_train = zeros(length(Y_train),K);
for i = 1:K
    rbf_train(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Z),1)-Z).^2,2)/(2*sigma^2));
end

%% Create training and testing sets

[trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);

X = [city_train(trainind,:) Z(trainind,:) rbf_train(trainind,:)];
Xtest = [city_train(testind,:) Z(testind,:) rbf_train(testind,:)];
Y = Y_train(trainind);
Ytest = Y_train(testind);
clusterIdsTrain = clusterIds(trainind);
clusterIdsTest = clusterIds(testind);

%% LASSO :D
if loading
    load('lassoWithRBF100_old.mat','Slope','Fitinfo');
else
    tic
    [Slope, Fitinfo] = lasso(X,Y);
    toc
end

%%
Slope0 = Slope(:,11);
Intercept0 = Fitinfo.Intercept(1,11);

Ztest = [word_test bigram_test] * V;
sigma = 10; % TODO: Tune this. Question: HOW? Lasso takes WAY too long.
rbf_test = zeros(size(Ztest,1),K);
for i = 1:K
    rbf_test(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Ztest),1)-Ztest).^2,2)/(2*sigma^2));
end
Xtest = [city_test Ztest rbf_test];

Yhat = Xtest*Slope0 + Intercept0;
dlmwrite('submit.txt',Yhat,'precision','%d');


