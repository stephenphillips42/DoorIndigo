%% Other kmeans attempt
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
Ztest = [word_test bigram_test] * V;
Zall = [Z;Ztest];

K = 200; % Number of clusters
loading = false;

%% Rerun kmeans
if loading
    load('kmeansStuff100_small.mat')
else
    tic
    clusterIds = kmeans(Zall,K);
    toc
end

%%
clusterMeans = zeros(K,size(Z,2));
clusterPrices = zeros(K,1);
for i = 1:K
    clusterMeans(i,:) = mean(Z(clusterIds==i,:));
    clusterPrices(i) = mean(Y_train(clusterIds==i));
end
save('kmeansStuff200.mat','clusterIds');

%
% figure;
% plot3(clusterMeans(:,1),clusterMeans(:,2),clusterPrices,'r.');

%%
K = 2000;
figure; hold on
plotpc1=1;
plotpc2=2;
plotpc3=500;
cc = hsv(K);
clusterStats = zeros(K,3);
for i = [1 247]
    clusterStats(i,:) = [ sum(clusterIds==i) mean(Y(clusterIds==i)) range(Y(clusterIds==i)) ];
    %fprintf('Cluster %d, with %d members, mean: %f, range %f\n',...
    %    i,clusterStats(i,1),clusterStats(i,2),clusterStats(i,3))
    scatter3(Z(clusterIds==i,plotpc1),...
          Z(clusterIds==i,plotpc2),...
          Z(clusterIds==i,plotpc3),...
          (Y(clusterIds==i)).^11/10000000000,...
          100*(Y(clusterIds==i)),...
          '.');
    % plot3(Z(clusterIds==i,plotpc1),Z(clusterIds==i,plotpc2),Z(clusterIds==i,plotpc3),'.','color',cc(i,:));
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

% [trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);

% X = [city_train Z rbf_train];
X = [rbf_train(trainind,:)];
Xtest = [rbf_train(testind,:)];
Y = Y_train;
% Ytest = Y_train(testind);
% clusterIdsTrain = clusterIds(trainind);
% clusterIdsTest = clusterIds(testind);

%% LASSO :D
% if loading
%     load('lassoWithRBF100_old.mat','Slope','Fitinfo');
% else
    tic
    [Slope, Fitinfo] = lasso(X,Y);
    toc
% end

%%
Slope0 = Slope(:,11);
Intercept0 = Fitinfo.Intercept(1,11);

% Ztest = [word_test bigram_test] * V;
% sigma = 10; % TODO: Tune this. Question: HOW? Lasso takes WAY too long.
% rbf_test = zeros(size(Ztest,1),K);
% for i = 1:K
%     rbf_test(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Ztest),1)-Ztest).^2,2)/(2*sigma^2));
% end
% Xtest = [city_test Ztest rbf_test];



Yhat = Xtest*Slope0 + Intercept0;
dlmwrite('submit.txt',Yhat,'precision','%d');


