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

initialize_additional_features;

%% Ridge regression
lambda = 1;

tic
Atb = word_train'*Y_train;
AtA = (word_train'*word_train + lambda*eye(nwords));
w_word = AtA \ Atb;
clear AtA Atb
toc

tic
Atb = bigram_train'*Y_train;
AtA = (bigram_train'*bigram_train + lambda*eye(nbigrams));
w_bigram = AtA \ Atb;
clear AtA Atb
toc

tic
Atb = [word_train bigram_train]'*Y_train;
AtA = ([word_train bigram_train]'*[word_train bigram_train] + lambda*eye(nwords+nbigrams));
w_both = AtA \ Atb;
clear AtA Atb
toc

%%
Yhat_word = word_train*w_word;
Yhat_bigram = bigram_train*w_bigram;
Yhat_both = [word_train bigram_train]*w_both;

err_word = sqrt(sum(Yhat_word-Y_train).^2/N);
err_bigram = sqrt(sum(Yhat_bigram-Y_train).^2/N);
err_both = sqrt(sum(Yhat_both-Y_train).^2/N);

plot(Y_train,Y_train-Yhat_word,'b.')
hold on
plot(Y_train,Y_train-Yhat_bigram,'r.')
plot(Y_train,Y_train-Yhat_both,'g.')
plot(Y_train,zeros(size(Y_train)),'k--','LineWidth',2)

%% Cross validation of lambdas
X = [word_train bigram_train];
K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K);
Lambdas = [0.5 1 10 100 1000 10000];
err_raw = zeros(length(Lambdas),K);

for l = 1:length(Lambdas)
    figure
    cc = hsv(K);
    for k = 1:K
        test = (Ind == k); train = ~test;
        tic
        Atb = X(train,:)'*Y_train(train,:);
        AtA = (X(train,:)'*X(train,:) + Lambdas(l)*eye(size(X(train,:),2)));
        w = AtA \ Atb;
        clear AtA Atb
        toc
        Yhat = X(test,:)*w;
        err_raw(l,k) = norm(Yhat - Y_train(test))/N;
        plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
        hold on
        plot(Y_train(test),zeros(size(Y_train(test))),'k.')
    end
    title(sprintf('Lambda = %f',Lambdas(l)))
end

fprintf('Optimal Lambda at 100\n')

%% PCA attempt
npcs = 200;
tic
[U,S,V]=svds(X_train,npcs);
Z = U*S;
toc

%%
% PCR
[N,npcs]=size(Z);
tic
lambda = 1;
AtA = Z'*Z;
Atb = Z'*Y_train;
w_pca = AtA \ Atb;
clear AtA b;
toc
Yhat = Z*w_pca;

err_pca = sqrt(sum((Y_train-Yhat).^2)/N);
fprintf('Error: %f\n',err_pca);
% Cross validation for PCA

% PCA K-fold cross validation
K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K);
err_pca = zeros(1,K); % Already regularized, do not need the lambdas
figure
cc = hsv(K);
for k = 1:K
    test = (Ind == k); train = ~test;
    tic
    Atb = Z(train,:)'*Y_train(train);
    AtA = Z(train,:)'*Z(train,:);
    w = AtA \ Atb;
    clear AtA Atb
    toc
    Yhat = Z(test,:)*w;
    err_pca(k) = norm(Yhat - Y_train(test))/length(Y_train(test));
    plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
    hold on
    plot(Y_train(test),zeros(size(Y_train(test))),'k.')
end
title(sprintf('PCA Regression (%d PCs)',npcs))

%[C,Score] = pca(X);
% for i = 1:length(C)
%     norm(Score(:,1:i)*coeff(1:i
% end

%% Data visualization
figure
plot3(Z((Y_city==1),1),Z((Y_city==1),2),Z((Y_city==1),3),'b.'); hold on
plot3(Z((Y_city==2),1),Z((Y_city==2),2),Z((Y_city==2),3),'r.')
plot3(Z((Y_city==3),1),Z((Y_city==3),2),Z((Y_city==3),3),'g.')
plot3(Z((Y_city==4),1),Z((Y_city==4),2),Z((Y_city==4),3),'m.')
plot3(Z((Y_city==5),1),Z((Y_city==5),2),Z((Y_city==5),3),'k.')
%%
figure
hold on
miny = 12.041;
maxy = 13.566;
plot3(Z((Y_train > miny & Y_train < maxy),1),Z((Y_train > miny & Y_train < maxy),2),Z((Y_train > miny & Y_train < maxy),3),'r.')
plot3(Z((Y_train > maxy),1),Z((Y_train > maxy),2),Z((Y_train > maxy),3),'g.')
plot3(Z((Y_train < miny),1),Z((Y_train < miny),2),Z((Y_train < miny),3),'b.');
%%
figure
hold on
miny = 12;
maxy = 13.566;
dimen = 2;
startdim = 1;
enddim = 75;
rng = enddim-startdim;
for i = startdim:enddim
    plot(3*i+zeros(sum(Y_train > miny & Y_train < maxy),1),Z((Y_train > miny & Y_train < maxy),i),'r.')
    plot(3*i+zeros(sum(Y_train > maxy),1)+1,Z((Y_train > maxy),i),'g.')
    plot(3*i+zeros(sum(Y_train < miny),1)+2,Z((Y_train < miny),i),'b.');
end
%%
figure
hold on
miny = 12;
maxy = 13.566;
dimens = [ 2 3 4 ];
plot3(Z((Y_train > miny & Y_train < maxy),2),Z((Y_train > miny & Y_train < maxy),3),Z((Y_train > miny & Y_train < maxy),4),'r.')
plot3(Z((Y_train > maxy),2),Z((Y_train > maxy),3),Z((Y_train > maxy),4),'g.')
plot3(Z((Y_train < miny),2),Z((Y_train < miny),3),Z((Y_train < miny),4),'b.');

%% GMM


