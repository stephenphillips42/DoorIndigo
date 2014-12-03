function Yhat = knnGaussianKernel( x, X, Y, sigma )
% KNNGAUSSIANKERNEL Apply K-NN to x using training data X and Y
% Input:
% x - nxP vector of the points we want to estimate
% X - NxP vector of training points to use K-NN on
% Y - Nx1 vector of the regression values of X
% Output:
% Yhat - scalar output of estimate

[~,p] = size(x);
[~,P] = size(X);

if p ~= P
    return
end

% Tricks with cells to apply to every row since rowfun only works on
% tables because it is dumb... >:(
Yhat = cell2mat(cellfun(@(x_) singleKNNGaussian(x_,X,Y,sigma),...
                    num2cell(x,2),'UniformOutput',false));

% Individual function is easy to define
function yhat = singleKNNGaussian(x,X,Y,sigma)
    knnGaussian = exp(-(sum((X-repmat(x,size(X,1),1)).^2,2)/sigma^2));
    if any(any(isnan(knnGaussian)))
        disp('ERROR')
    end
    yhat = sum(knnGaussian.*Y)/(eps+sum(knnGaussian));
end







end

