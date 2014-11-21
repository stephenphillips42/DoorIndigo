%% Trying pure PCR
[N,npcs]=size(Z);
% Getting the weights
tic
lambda = 1;
AtA = Z'*Z;
Atb = Z'*Y_train;
w_pca = AtA \ Atb;
clear AtA Atb; % Memory issues on my computer require this
toc

Yhat = Z*w_pca;

err_pca = sqrt(sum((Y_train-Yhat).^2)/N);
fprintf('Error: %f\n',err_pca);

% PCA K-fold cross validation

K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K); % Hooray MATLAB function
err_pca = zeros(1,K); % Already regularized, do not need the lambdas
figure
cc = hsv(K); % Colors for different test sets
for k = 1:K
    % Get indecies
    test = (Ind == k);
    train = ~test;
    
    % Generate weights
    tic
    Atb = Z(train,:)'*Y_train(train);
    AtA = Z(train,:)'*Z(train,:);
    w = AtA \ Atb;
    clear AtA Atb
    toc
    
    % Test error
    Yhat = Z(test,:)*w;
    err_pca(k) = norm(Yhat - Y_train(test))/length(Y_train(test));

    % Plot error
    plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
    hold on
    plot(Y_train(test),zeros(size(Y_train(test))),'k.')
end
title(sprintf('PCA Regression (%d PCs)',npcs))


%% GMM
% gobj = fitgmdist(Z_all,1);
% gobj2 = fitgmdist(Z_all,2);
% gobj3 = fitgmdist(Z_all,3);
%idx = cluster(gobj2,Z);

% This crashes on occasion and I don't know why
figure
hold on
miny = 12;
maxy = 13.566;
dimens = [ 2 3 4 ];
plot3(Z((idx==1),dimens(1)),Z((idx==1),dimens(2)),Z((idx==1),dimens(3)),'b.')
plot3(Z((idx==2),dimens(1)),Z((idx==2),dimens(2)),Z((idx==2),dimens(3)),'r.');
% plot3(Z((idx==3),dimens(1)),Z((idx==3),dimens(2)),Z((idx==3),dimens(3)),'g.');


%% Decision trees

leafs = linspace(150,300,10);
% Testing for optimal tree height (for MATLAB function specified
%  using 'MinLeaf')
% Optimal leaf structure not clear... around 215

rng('default');
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    disp(leafs(n))
    t = fitctree(Z,(Y_train<prctile(Y_train,25)),'CrossVal','On',...
        'MinLeaf',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');


%%
X = [Z word_train bigram_train];
K = 6; % Number of cross validations
Ind = crossvalind('Kfold',size(X,1),K);
Lambdas = [0.1];

err_dtree = zeros(length(Lambdas),K);

for l = 1:length(Lambdas)
    disp(Lambdas(l))
    figure
    cc = hsv(K);
    for k = 1:K
        test = (Ind == k); train = ~test;
        t25 = fitctree(Z(train,:),...
                        (Y_train(train)<prctile(Y_train(train),25)),...
                        'MinLeaf',215);
        t75 = fitctree(Z(train,:),...
                        (Y_train(train)<prctile(Y_train(train),75)),...
                        'MinLeaf',215);
        Xtmp = X(train,:);
        Ytmp = Y_train(train);
        Y25 = predict(t25,Z(train,:));
        Y75 = predict(t75,Z(train,:));
        % Group 1 in 0-25% range
        tic
        disp('Group 1 - <25')
        Atb = Xtmp(Y25,:)'*Ytmp(Y25);
        AtA = (Xtmp(Y25,:)'*Xtmp(Y25,:));
        for i = 1:size(AtA,1)
            AtA(i,i) = AtA(i,i)+Lambdas(l);
        end
        w25 = AtA \ Atb;
        clear AtA Atb
        toc
        % Group 2 in 25-75% range
        disp('Group 2 - 25-75')
        tic
        Atb = Xtmp(~Y25 & Y75,:)'*Ytmp(~Y25 & Y75);
        AtA = (Xtmp(~Y25 & Y75,:)'*Xtmp(~Y25 & Y75,:));
        for i = 1:size(AtA,1)
            AtA(i,i) = AtA(i,i)+Lambdas(l);
        end
        wmid = AtA \ Atb;
        clear AtA Atb
        toc
        % Group 3 in 75-100% range (minus some mis-classifications)
        disp('Group 3 - >75')
        tic
        Atb = Xtmp(~Y25 & ~Y75,:)'*Ytmp(~Y25 & ~Y75);
        AtA = (Xtmp(~Y25 & ~Y75,:)'*Xtmp(~Y25 & ~Y75,:));
        for i = 1:size(AtA,1)
            AtA(i,i) = AtA(i,i)+Lambdas(l);
        end
        w75 = AtA \ Atb;
        clear AtA Atb
        toc
        clear Xtmp Ytmp

        % Predictions
        Yhat = zeros(size(Y_train(test)));
        % tree predictions
        Ytest25 = predict(t25,Z(test,:));
        Ytest75 = predict(t75,Z(test,:));
        % Regression from trees
        Xtmp_test = X(test,:);
        disp('Testing <25')
        Yhat(Ytest25) = Xtmp_test(Ytest25,:)*w25;
        disp('Testing 25-75')
        Yhat(~Ytest25 & Ytest75) = Xtmp_test(~Ytest25 & Ytest75,:)*wmid;
        disp('Testing >75')
        Yhat(~Ytest25 & ~Ytest75) = Xtmp_test(~Ytest25 & ~Ytest75,:)*w75;
        
        clear Ytest25 Ytest75 Xtmp_test
        
        % Record results
        err_dtree(l,k) = norm(Yhat - Y_train(test))/length(Y_train(test));
        plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
        hold on
        plot(Y_train(test),zeros(size(Y_train(test))),'k.')
    end
    title(sprintf('Lambda = %f',Lambdas(l)))
end

disp(sum(err_dtree,2))

