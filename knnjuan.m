function knnjuan(vecinos,trainImages,trainLabels,testImages,testLabels)

%Inteligencia Artificial Aplicada
%Proyecto: KNN

MdlKnn = fitcknn(trainImages',trainLabels','NumNeighbors',vecinos);

testKnn=testImages';
labelKnn = predict(MdlKnn,testKnn);
labelKnn=labelKnn';
errKnn=length(find(labelKnn~=testLabels));

disp('Número de vecinos:')
disp(vecinos);
disp('Error knn:')
disp(errKnn);

%confusion matrix knn:
figure
cmKnn = confusionchart(testLabels',labelKnn);
cmKnn.Title = 'Confusion Matrix KNN';
end