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


%% Cross validation of lambdas
X = [word_train bigram_train];
K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K);
Lambdas = [80 90 100 200 500];
err_raw = zeros(length(Lambdas),K);

for l = 1:length(Lambdas)
    figure
    cc = hsv(K);
    for k = 1:K
        test = (Ind == k); train = ~test;
        tic
        Atb = X(train,:)'*Y_train(train,:);
        AtA = (X(train,:)'*X(train,:));
        for i = 1:size(AtA,1)
            AtA(i,i) = AtA(i,i)+Lambdas(l);
        end
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

%fprintf('Optimal Lambda at 100\n')

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
dimens = [ 1 6 6 ];
plot3(Z((Y_train > miny & Y_train < maxy),dimens(1)),Z((Y_train > miny & Y_train < maxy),dimens(2)),Z((Y_train > miny & Y_train < maxy),dimens(3)),'b.')
plot3(Z((Y_train > maxy),dimens(1)),Z((Y_train > maxy),dimens(2)),Z((Y_train > maxy),dimens(3)),'g.')
plot3(Z((Y_train < miny),dimens(1)),Z((Y_train < miny),dimens(2)),Z((Y_train < miny),dimens(3)),'r.');

%% GMM
% gobj = fitgmdist(Z_all,1);
% gobj2 = fitgmdist(Z_all,2);
% gobj3 = fitgmdist(Z_all,3);
%idx = cluster(gobj2,Z);

figure
hold on
miny = 12;
maxy = 13.566;
dimens = [ 2 3 4 ];
plot3(Z((idx==1),dimens(1)),Z((idx==1),dimens(2)),Z((idx==1),dimens(3)),'b.')
plot3(Z((idx==2),dimens(1)),Z((idx==2),dimens(2)),Z((idx==2),dimens(3)),'r.');
% plot3(Z((idx==3),dimens(1)),Z((idx==3),dimens(2)),Z((idx==3),dimens(3)),'g.');
% Conclusion: Not very helpful

%%




