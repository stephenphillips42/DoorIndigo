%% Kmeans
K = 100;
%%
Zprime = Z;
Z = Zprime(:,1:500);
%% Rerun kmeans
tic
clusterIds = kmeans(Z,K);
toc
%% Run GMM

% fitgmdist(X,k) Never mind this would take way too long

%%
clusterMeans = zeros(K,size(Z,2));
clusterPrices = zeros(K,1);
for i = 1:K
    clusterMeans(i,:) = mean(Z(clusterIds==i,:));
    clusterPrices(i) = mean(Y_train(clusterIds==i));
end
save('kmeansStuff5.mat','Z','clusterMeans','clusterIds','clusterPrices');

%% Load Kmeans
%load('kmeansStuff20.mat')

figure;
plot3(clusterMeans(:,1),clusterMeans(:,2),clusterPrices,'r.');
%%
sample = unidrnd(100,length(Y_train),1);
scatter3(Z(sample,plotpc1),...
          Z(sample,plotpc2),...
          Z(sample,plotpc3),...
          (Y_train(sample)).^11/10^10,...
          100*(Y_train(sample)),...
          '.');
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

%% Radial basis functions with the means
% Compute the feature vectors
sigma = 10; % TODO: Tune this. Question: HOW? Lasso takes WAY too long.
rbf_train = zeros(length(Y_train),K);
for i = 1:K
    rbf_train(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Z),1)-Z).^2,2)/(2*sigma^2));
end
% 
% for i = unidrnd(K,10,1)
%     figure(3)
%     hist(rbf_train(:,i))
%     title(sprintf('RBF %d',i))
%     pause(1)
% end
mean(std(rbf_train,[],2))
plot(std(rbf_train,[],1))

%% Create training and testing sets
[trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);
X = [city_train(trainind,:) Z(trainind,:) rbf_train(trainind,:)];
Xtest = [city_train(testind,:) Z(testind,:) rbf_train(testind,:)];
Y = Y_train(trainind);
Ytest = Y_train(testind);
clusterIdsTrain = clusterIds(trainind);
clusterIdsTest = clusterIds(testind);

%% LASSO :D
tic
[Slope, Fitinfo] = lasso(X,Y);
toc

save('lassoWithRBF100_all.mat','Slope','Fitinfo');


%%
Slope0 = Slope(:,11);
Intercept0 = Fitinfo.Intercept(1,11);
Yhat = Xtest*Slope0 + Intercept0;

norm(Yhat-Ytest)/sqrt(length(Ytest))
plot(Ytest, Ytest-Yhat,'r.')
hold on
plot(Ytest,zeros(size(Ytest)),'b.')
hold off


