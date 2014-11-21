clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X_train =[city_train word_train bigram_train];
Y_train = price_train;
X_test = [city_test word_test bigram_test];

initialize_additional_features;

%% Run algorithm
% Example by lazy TAs
% [N,npcs]=size(Z);
% X = [Z word_train bigram_train];
% 
% tic
% lambda = 0.1;
% AtA = X'*X;
% for i = 1:size(AtA,1)
%     AtA(i,i) = AtA(i,i)+lambda;
% end
% Atb = X'*Y_train;
% w_ridge = AtA \ Atb;
% clear AtA Atb;
% toc
%%
X = [Z word_train bigram_train];
w_lasso = lasso(X,Y_train,'Options','UseParallel');

%%
%prices = full([Z_test word_test bigram_test])*w_lasso;

%% Save results to a text file for submission
%dlmwrite('submit.txt',prices,'precision','%d');