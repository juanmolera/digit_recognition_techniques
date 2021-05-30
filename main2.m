clear all
close all
clc

%Main2

tic;
load('Trainnumbers.mat');

%K-means:
%Para determinar la k:

% numClusters=[10:10:200];
% errVector=[];
% 
% for i=1:length(numClusters)
%     [errorGlobal]=kmeansjuan(numClusters(i));
%     errVector=horzcat(errVector,errorGlobal);
% end
% 
% plot(numClusters,errVector,'r-',numClusters,errVector,'r*')
% xlabel('Número de Clústers')
% ylabel('Error global K-means')

%Usando la k:
numClusters=100;
[errorGlobal]=kmeansjuan(numClusters);
disp('Train error:')
disp(errorGlobal);

test_images = Trainnumbers.image(:,8001:10000);
test_labels = Trainnumbers.label(:,8001:10000);
l = load('centroidLabels');
cLabels = l.centroidLabels;
c = load('centroids');
savedCent = c.centroids;

[~,idx_test] = pdist2(savedCent',test_images','euclidean','Smallest',1);
for j=1:length(idx_test)
    for k=1:length(cLabels)
        if k==idx_test(j)
            idx_test(j)=cLabels(k);
        end
    end   
end

errorTest=length(find(idx_test~=test_labels))/10000;
disp('Test error:')
disp(errorTest);

toc;