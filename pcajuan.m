function [pca]=pcajuan(ncomp,trainImages)

%Inteligencia Artificial Aplicada
%Proyecto: PCA

%mean, std y train data normalized
meanImages=mean(trainImages')'; 
stdImages=std(trainImages')';
normalized=(zscore(trainImages'))';

%cálculo de eigenvectors y eigenvalues
[eigvectors,eigvalues]=eig(cov(normalized'));

%vector con eigenvalues
for n=1:784
    eigvaluesVector(1,n)=eigvalues(n,n);
end

%ordenado de mayor a menor
[sorted,ind]=sort(eigvaluesVector,'descend');

%me quedo con los X primeros eigvectors
%para 10 primeros:
%ncomp=19;
disp('Número de componentes:')
disp(ncomp);

for m=1:ncomp
    engienvectorXd(:,m)=eigvectors(:,ind(1,m));
end

%proyección
pcaNormalized=((engienvectorXd'*(normalized))'*engienvectorXd')';

for i=1:length(trainImages)
    %784x1,784x10k,784x1,784x1
    aux=pcaNormalized(:,i).*stdImages+meanImages;
    for j=1:784
        %Valores inferiores a 100 se consideran 0
        if aux(j)<0
            pca(j,i)=0;
        else
            pca(j,i)=aux(j);
        end
    end  
end

% list=[42,46];
% for z=1:length(list)
%     k=list(z);
%     for i=1:28
%         for j=1:28
%             digit(i,j)=pca((i-1)*28+j,k);
%         end
%     end
%     %figure;
%     %imshow(digit);  
% end

%cálculo de MSE:
err=immse(trainImages,pca);
errn=immse(normalized,pcaNormalized);
disp('Error:')
disp(err);
disp('Error normalizado:')
disp(errn);
end
