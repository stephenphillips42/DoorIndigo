% Ridge Regression Analysis
% 
% Showed that using both words and bigrams is the better option, but that
% words alone are more discriminative

%% Ridge regression
lambda = 1;

nwords = size(word_train,2);
nbigrams = size(bigram_train,2);

% Memory squeeze
clear X_test

tic
Atb = word_train'*Y_train;
AtA = (word_train'*word_train);
% Memory saver
for i = 1:size(AtA,1)
    AtA(i,i) = AtA(i,i)+lambda;
end
w_word = AtA \ Atb;
clear AtA Atb
toc

tic
Atb = bigram_train'*Y_train;
AtA = (bigram_train'*bigram_train);
for i = 1:size(AtA,1)
    AtA(i,i) = AtA(i,i)+lambda;
end
w_bigram = AtA \ Atb;
clear AtA Atb
toc

tic
Atb = [word_train bigram_train]'*Y_train;
AtA = ([word_train bigram_train]'*[word_train bigram_train]);
for i = 1:size(AtA,1)
    AtA(i,i) = AtA(i,i)+lambda;
end
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
