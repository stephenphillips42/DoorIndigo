function prediction = make_final_prediction(model,X_test)

% Input
% X_test : a 1xp vector representing "1" test sample.
% X_test=[city word bigram] a 1-by-10007 vector
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

