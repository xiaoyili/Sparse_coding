function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
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

    isTopo = 1;
%     L = size(groupMatrix,1);
%     [K M] = size(featureMatrix);
    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
        if(isequal(groupMatrix,eye(numFeatures)));
            isTopo = 0;
        end
    else
        groupMatrix = eye(numFeatures);
         isTopo = 0;
    end
    
    numExamples = size(patches, 2);
    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
    
    
    %% 求目标的代价函数
    delta = weightMatrix*featureMatrix-patches;
    fResidue = sum(sum(delta.^2))./numExamples;%重构误差
%     fWeight = gamma*sum(sum(weightMatrix.^2));%防止基内元素值过大
    sparsityMatrix = sqrt(groupMatrix*(featureMatrix.^2)+epsilon);
    fSparsity = lambda*sum(sparsityMatrix(:)); %对特征系数性的惩罚值
%     cost = fResidue++fSparsity+fWeight;%此时A为常量，可以不用
    cost = fResidue++fSparsity;

    %% 求目标代价函数的偏导函数
    gradResidue = (-2*weightMatrix'*patches+2*weightMatrix'*weightMatrix*featureMatrix)./numExamples;
  
    % 非拓扑结构时：
    if ~isTopo
        gradSparsity = lambda*(featureMatrix./sparsityMatrix);
    end
    
    % 拓扑结构时
    if isTopo
%         gradSparsity = lambda*groupMatrix'*(groupMatrix*(featureMatrix.^2)+epsilon).^(-0.5).*featureMatrix;%一定要小心最后一项是内积乘法
        gradSparsity = lambda*groupMatrix'*(groupMatrix*(featureMatrix.^2)+epsilon).^(-0.5).*featureMatrix;%一定要小心最后一项是内积乘法
    end
    grad = gradResidue+gradSparsity;
    grad = grad(:);
    
end
