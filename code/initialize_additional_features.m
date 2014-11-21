% This is the script to process additional features that you want to use

% YOUR CODE GOES HERE

% load ../data/metadata.mat

X_train_additional_features = []; % Modify this in if needed
X_test_additional_features = []; % Modify this in if needed


npcs = 200;
fprintf('PCA computation ...\n')
tic
[U,S,V]=svds([X_train; X_test],npcs);
Z_all = U*S;
Z = Z_all(1:size(X_train,1),:);
Z_test = Z_all((1+size(X_train,1)):end,:);

toc



