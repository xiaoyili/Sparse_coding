function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);%非拓扑的sparse coding中，相当于groupMatrix为单位对角矩阵
    end

    numExamples = size(patches, 2);%测试代码时为5

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);%其实传入进来的就是这些东西
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------    
    %% 求目标的代价函数
    delta = weightMatrix*featureMatrix-patches;
    fResidue = sum(sum(delta.^2))./numExamples;%重构误差
    fWeight = gamma*sum(sum(weightMatrix.^2));%防止基内元素值过大
%     sparsityMatrix = sqrt(groupMatrix*(featureMatrix.^2)+epsilon);
%     fSparsity = lambda*sum(sparsityMatrix(:)); %对特征系数性的惩罚值
%     cost = fResidue+fWeight+fSparsity; %目标的代价函数
    cost = fResidue+fWeight;
    
    %% 求目标代价函数的偏导函数
    grad = (2*weightMatrix*featureMatrix*featureMatrix'-2*patches*featureMatrix')./numExamples+2*gamma*weightMatrix;
    grad = grad(:);
   
end
