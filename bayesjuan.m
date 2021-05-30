clear all
close all
clc

%Inteligencia Artificial Aplicada
%Proyecto: Naive-Bayes
tic;
load('Trainnumbers.mat');

%pca
[~,pcaImages,~]=pca(Trainnumbers.image','NumComponents',60);

%separar train-test
%80-20
trainImages=pcaImages(1:8000,:);
trainLabels=Trainnumbers.label(1:8000);
testImages=pcaImages(8001:10000,:);
testLabels=Trainnumbers.label(8001:10000);

%entrenar modelo
Mdl = fitcnb(trainImages,categorical(trainLabels)');

%test
y1 = predict(Mdl,testImages);
y1_c=categorical(y1');
t_c=categorical(testLabels);

%confusion matrix
figure;
plotconfusion(y1_c,t_c);
toc;