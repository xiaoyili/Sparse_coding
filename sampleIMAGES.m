function patches = sampleIMAGES(images, patchsize,numpatches)
% sampleIMAGES
% Returns 10000 patches for training

% load IMAGES;    % load images from disk 

%patchsize = 8;  % we'll use 8x8 patches 
%numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
for imageNum = 1:10%在每张图片中随机选取1000个patch，共10000个patch
    [rowNum colNum] = size(images(:,:,imageNum));
    for patchNum = 1:2000%实现每张图片选取1000个patch
        xPos = randi([1,rowNum-patchsize+1]);
        yPos = randi([1, colNum-patchsize+1]);
        patches(:,(imageNum-1)*2000+patchNum) = reshape(images(xPos:xPos+patchsize-1,yPos:yPos+patchsize-1,...
                                                        imageNum),patchsize*patchsize,1);
    end
end


%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
% patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;%因为根据3sigma法则，95%以上的数据都在该区域内
                                                % 这里转换后将数据变到了-1到1之间

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
