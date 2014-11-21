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


%% Kmeans
X = [Z word_train bigram_train];
Idx = kmeans(X,30);


%% Cross validation of Experiment (template)
K = 10; % Number of cross validations
Ind = crossvalind('Kfold',N,K);
Lambdas = [0.1 10 50 80]; % Different regularization parameters we choose
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
        err_raw(l,k) = norm(Yhat - Y_train(test))/length(Y_train(test));
        plot(Y_train(test),Y_train(test)-Yhat,'.','color',cc(k,:))
        hold on
        plot(Y_train(test),zeros(size(Y_train(test))),'k.')
    end
    title(sprintf('Lambda = %f',Lambdas(l)))
end

%fprintf('Optimal Lambda at 100\n')





