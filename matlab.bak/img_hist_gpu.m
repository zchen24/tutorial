% Shows how to use gpuArray image processing
% Zihan Chen
% Date: 2020-01-04


close all; clear; clc;

load('mristack');

im1 = mristack(:,:,1);
figure; imshow(im1); title('mri image 1');


im1_histeq = histeq(im1);
figure; imshow(im1_histeq); title('mri image 1 with histeq');

g_im1 = gpuArray(im1);
g_im1_histeq = histeq(g_im1);
figure; imshow(g_im1_histeq); title('mri image 1 with histeq GPU');


% manually do it
num_bins = 256;
[cnts, bins] = imhist(im1, num_bins);

% cumsum()
cdf = cumsum(cnts);
cdf = 256 * cdf / cdf(end);


% sample points: bins
% sample value: cdf
% query value: im1
im1_eq = interp1(bins, cdf, single(im1(:))); 
im1_eq = reshape(im1_eq, size(im1));
im1_eq = uint8(im1_eq);  % convert back to uint8
figure; imshow(im1_eq); title('mri image 1 with manual hist eq');