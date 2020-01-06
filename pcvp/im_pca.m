% Image PCA analysis
% See PCVP P14 PCA of Images
% fontimages.zip can be downloaded from 
% http://programmingcomputervision.com/
%
% 2020-01-05

clc; clear; close all;

folder = './pcv_data/a_thumbs/';
imlist = dir([folder '*.jpg']);

im1 = imread(fullfile(folder, imlist(1).name));

immatrix = zeros(length(im1(:)), length(imlist));
for i = 1:length(imlist)
	tmp = imread(fullfile(folder, imlist(i).name));
	immatrix(:,i) = tmp(:);
end

immatrix = double(immatrix');
[coeffs, ~, latent, ~, explained, mu] = pca(immatrix, 'NumComponents', 7);


% or manually
% X = immatrix - repmat(mean(immatrix),size(immatrix,1), 1);
% [U, S, V] = svd(X);
% coeffs = V(:,7);


figure;
subplot(2, 4, 1);
imshow(reshape(uint8(mu), size(im1)));
for i = 1:7
	subplot(2, 4, i+1);
	imshow(reshape(coeffs(:,i)+0.5, size(im1)));
end

imshow(reshape(uint8(mu), size(im1)));