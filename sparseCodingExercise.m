%% CS294A/CS294W Sparse Coding Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sparse coding exercise. In this exercise, you will need to modify
%  sparseCodingFeatureCost.m and sparseCodingWeightCost.m. You will also
%  need to modify this file, sparseCodingExercise.m slightly.

% Add the paths to your earlier exercises if necessary
% addpath /path/to/solution

%%======================================================================
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

addpath minFunc;
numPatches = 20000;   % number of patches
numFeatures = 256;    % number of features to learn
patchDim = 16;         % patch dimension
visibleSize = patchDim * patchDim; %单通道灰度图，64维，学习121个特征

% dimension of the grouping region (poolDim x poolDim) for topographic sparse coding
poolDim = 3;

% number of patches per batch
batchNumPatches = 2000; %分成10个batch

lambda = 5e-5;  % L1-regularisation parameter (on features)
epsilon = 1e-5; % L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
gamma = 1e-2;   % L2-regularisation parameter (on basis)

%%======================================================================
%% STEP 1: Sample patches

images = load('IMAGES.mat');
images = images.IMAGES;
patches = sampleIMAGES(images, patchDim, numPatches);
display_network(patches(:, 1:64));

%%======================================================================
%% STEP 3: Iterative optimization
%  Once you have implemented the cost functions, you can now optimize for
%  the objective iteratively. The code to do the iterative optimization 
%  using mini-batching and good initialization of the features has already
%  been included for you. 
% 
%  However, you will still need to derive and fill in the analytic solution 
%  for optimizing the weight matrix given the features. 
%  Derive the solution and implement it in the code below, verify the
%  gradient as described in the instructions below, and then run the
%  iterative optimization.

% Initialize options for minFunc
options.Method = 'cg';
options.display = 'off';
options.verbose = 0;

% Initialize matrices
weightMatrix = rand(visibleSize, numFeatures);%64*121
featureMatrix = rand(numFeatures, batchNumPatches);%121*2000

% Initialize grouping matrix
assert(floor(sqrt(numFeatures)) ^2 == numFeatures, 'numFeatures should be a perfect square');
donutDim = floor(sqrt(numFeatures));
assert(donutDim * donutDim == numFeatures,'donutDim^2 must be equal to numFeatures');

groupMatrix = zeros(numFeatures, donutDim, donutDim);%121*11*11
groupNum = 1;
for row = 1:donutDim
    for col = 1:donutDim 
        groupMatrix(groupNum, 1:poolDim, 1:poolDim) = 1;%poolDim=3
        groupNum = groupNum + 1;
        groupMatrix = circshift(groupMatrix, [0 0 -1]);
    end
    groupMatrix = circshift(groupMatrix, [0 -1, 0]);
end
groupMatrix = reshape(groupMatrix, numFeatures, numFeatures);%121*121

if isequal(questdlg('Initialize grouping matrix for topographic or non-topographic sparse coding?', 'Topographic/non-topographic?', 'Non-topographic', 'Topographic', 'Non-topographic'), 'Non-topographic')
    groupMatrix = eye(numFeatures);%非拓扑结构时的groupMatrix矩阵
end

% Initial batch
indices = randperm(numPatches);%1*20000
indices = indices(1:batchNumPatches);%1*2000
batchPatches = patches(:, indices);                           

fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fSparsity','fWeight');
warning off;
for iteration = 1:200   
  %  iteration = 1;
    error = weightMatrix * featureMatrix - batchPatches;
    error = sum(error(:) .^ 2) / batchNumPatches;  %说明重构误差需要考虑样本数
    fResidue = error;
    num_batches = size(batchPatches,2);
    R = groupMatrix * (featureMatrix .^ 2);
    R = sqrt(R + epsilon);    
    fSparsity = lambda * sum(R(:));    %稀疏项和权值惩罚项不需要考虑样本数
    
    fWeight = gamma * sum(weightMatrix(:) .^ 2);
    
    %上面的那些权值都是随机初始化的
    fprintf('  %4d  %10.4f  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fSparsity+fWeight, fResidue, fSparsity, fWeight)
               
    % Select a new batch
    indices = randperm(numPatches);
    indices = indices(1:batchNumPatches);
    batchPatches = patches(:, indices);                    
    
    % Reinitialize featureMatrix with respect to the new
    % 对featureMatrix重新初始化，按照网页教程上介绍的方法进行
    featureMatrix = weightMatrix' * batchPatches;
    normWM = sum(weightMatrix .^ 2)';
    featureMatrix = bsxfun(@rdivide, featureMatrix, normWM); 
    
    % Optimize for feature matrix    
    options.maxIter = 20;
    %给定权值初始值，优化特征值矩阵
    [featureMatrix, cost] = minFunc( @(x) sparseCodingFeatureCost(weightMatrix, x, visibleSize, numFeatures, batchPatches, gamma, lambda, epsilon, groupMatrix), ...
                                           featureMatrix(:), options);
    featureMatrix = reshape(featureMatrix, numFeatures, batchNumPatches);                                      
    weightMatrix = (batchPatches*featureMatrix')/(gamma*num_batches*eye(size(featureMatrix,1))+featureMatrix*featureMatrix');
    [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures, batchPatches, gamma, lambda, epsilon, groupMatrix);
          
end
    figure;
    display_network(weightMatrix);
