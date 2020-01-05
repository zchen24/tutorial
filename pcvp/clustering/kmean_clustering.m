% Shows how to use k-means clustering
% 
% 2020-01-04

clc; clear;

class1 = randn(100, 2);
class2 = randn(100, 2) + [5 5];
features = [class1; class2];

K = 2;
% IDX: index for each feature
%   C: centroids for each class
[IDX, C] = kmeans(features, K);

histogram(class1);

plot(features(:,1), features(:,2), '*');
xlim([-4 10]); ylim([-4 10]); grid on;

figure;
idx1 = features(IDX==1,:);
idx2 = features(IDX==2,:);
plot(idx1(:,1), idx1(:,2), 'r*');
hold on;
plot(idx2(:,1), idx2(:,2), 'k.');
% centroids
plot(C(:,1), C(:,2), 'go', 'MarkerFaceColor', 'g');
grid on;



