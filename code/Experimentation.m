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




%% Cross validation of Experiment (template)
X=[Z word_train];
K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K);
Lambdas = [0.1]; % Different regularization parameters we choose
err_raw = zeros(length(Lambdas),K);

% Optimal lambda for simple ridge regression (just words and bigrams) is 70
% Optimal lambda for extended ridge (using PCA as well) is 0.1 (zero makes
%   it crash??)

for l = 1:length(Lambdas)
    disp(Lambdas(l))
    figure
    cc = hsv(K);
    for k = 1:K
        test = (Ind == k); train = ~test;
        tic % Regression
        Atb = X(train,:)'*Y_train(train,:);
        AtA = (X(train,:)'*X(train,:));
        for i = 1:size(AtA,1)
            AtA(i,i) = AtA(i,i)+Lambdas(l);
        end
        w = AtA \ Atb;
        clear AtA Atb
        toc
        Yhat = X(test,:)*w;
        err_raw(l,k) = norm(Yhat - Y_train(test))/length(Y_train(test));
        plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
        hold on
        plot(Y_train(test),zeros(size(Y_train(test))),'k.')
    end
    title(sprintf('Y error vs Y. Lambda = %f',Lambdas(l)))
end

%fprintf('Optimal Lambda at 100\n')

%% SVM for large ones (too large to save)
[trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);

bigLabels = zeros(size(Y_train));
bigLabels(Y_train > 15) = 2;
bigLabels(bigLabels ~= 2) = 1;

bigLabels_train = bigLabels(trainind);
bigLabels_test = bigLabels(testind);
X = Z(trainind,1:500);
Xtest = Z(testind,1:500);

tic
bigSvm = svmtrain(X,bigLabels_train,'kernel_function','rbf','rbf_sigma',10);
toc

bigLabelHat = svmclassify(bigSvm,Xtest);

fprintf('Classification error by small SVM: %f%%\n',100*sum(bigLabelHat ~= bigLabels_test)/length(bigLabels_test));

%% SVM Training
[trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);

% SVM output as feature
smallLabels = zeros(size(Y_train));
smallLabels(Y_train < 10) = 1;
smallLabels(smallLabels ~= 1) = 2;

smallLabels_train = smallLabels(trainind);
smallLabels_test = smallLabels(testind);
X = Z(trainind,1:500);
Xtest = Z(testind,1:500);
tic
smallSvm = svmtrain(X,smallLabels_train,'kernel_function','rbf','rbf_sigma',10);
toc
smallLabelHat = svmclassify(smallSvm,Xtest);

fprintf('Classification error by small SVM: %f%%\n',100*sum(smallLabelHat ~= smallLabels_test)/length(smallLabels_test));

%% Create training and testing sets

%[trainind, testind] = crossvalind('HoldOut', length(Y_train), 0.5);
trainind = 1:length(Y_train);
Y = Y_train(trainind);
% Ytest = Y_train(testind);

% Compute the feature vectors
tic
sigma = 8; % TODO: Tune this. Question: HOW? Lasso takes WAY too long.
rbf_train = zeros(length(Y_train),K+1);
for i = 1:K
    rbf_train(:,i) = exp(-sum((repmat(clusterMeans(i,:),length(Z),1)-Z).^2,2)/(2*sigma^2));
end

smallLabelTrain = svmclassify(smallSvm,Z(trainind,1:500));
% smallLabelTest = svmclassify(smallSvm,Z(testind,1:500));
smallLabelTrain = 2*(smallLabelTrain-1.5);
% smallLabelTest = 2*(smallLabelTest-1.5);

npcs=500;
interinds = 1:20;
interterms = [ (interinds)' (interinds)'; nchoosek(interinds,2) ];
Zinter = Z(:,interterms(:,1)) .* Z(:,interterms(:,2));
Ztrain = Z(trainind,1:npcs);
% Ztest = Z(testind,1:npcs);
Zintertrain = Zinter(trainind,:);
% Zintertest = Zinter(testind,:);
X = full([...
     city_train(trainind,:) ...
     Ztrain ...
     Zintertrain ...
     rbf_train(trainind,:) ...
     word_train(trainind,wordsel(1:300)) ...
     bigram_train(trainind,bigramsel(1:200)) ...
     smallLabelTrain ]);
% Xtest = full([...
%          city_train(testind,:) ...
%          Ztest ...
%          Zintertest ...
%          rbf_train(testind,:) ...
%          word_train(testind,wordsel(1:300)) ...
%          bigram_train(testind,bigramsel(1:200)) ...
%          smallLabelTest ]);
toc
size(X)

% w = (Ztrain'*Ztrain + 1*eye(size(Ztrain,2))) \ Ztrain'*Y;
tic
% Next up: Lambda of 0.0001!!
[w, Fitinfo] = lasso(X,Y,'Lambda',0.01,'Alpha',0.4);
toc
b = Fitinfo.Intercept(1);
Yhat = (Xtest*w + b);
% disp(norm(Ytest-Yhat)/sqrt(length(Ytest)))
%%
plot(Ytest,Yhat,'r.');
hold on;
plot(Ytest,Ytest,'b.');
axis image

%%
figure
plot(1:size(w,1),w,'b-','LineWidth',3)

[sel, selInd] = sort(abs(w),'descend');


%%
tic

npcs=500;
interinds = 1:20;
interterms = [ (interinds)' (interinds)'; nchoosek(interinds,2) ];

Ztest = [word_test bigram_test]*V(:,1:npcs);
Zintertest = Ztest(:,interterms(:,1)) .* Ztest(:,interterms(:,2));

smallLabelTest = svmclassify(smallSvm,Ztest);
smallLabelTest = 2*(smallLabelTest-1.5);

sigma = 8;
rbf_test = zeros(size(Ztest,1),size(clusterMeans,1)+1);
for i = 1:K
    rbf_test(:,i) = exp(-sum((repmat(clusterMeans(i,1:500),size(Ztest,1),1)-Ztest).^2,2)/(2*sigma^2));
end

Xtest = full([...
         city_test ...
         Ztest ...
         Zintertest ...
         rbf_test ...
         word_test(:,wordsel(1:300)) ...
         bigram_test(:,bigramsel(1:200)) ...
         smallLabelTest ]);
toc

Yhat = (Xtest*w + b);

dlmwrite('submit.txt',Yhat,'precision','%d');


%% Unsupervised Neural Net (I know...)





