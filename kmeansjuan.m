function [errorGlobal]=kmeansjuan(k)

%Inteligencia Artificial Aplicada
%Proyecto: k-means
%tic;
load('Trainnumbers.mat');

a=Trainnumbers.image(:,1:8000);
trainLabels=Trainnumbers.label(:,1:8000);
% a=Trainnumbers.image;
% trainLabels=Trainnumbers.label;

%k=4; %Número de clústers
b = cast(a,'double');
c=b./255;
[class,cen]=kmeans(c',k);
%class: a qué clúster pertenece la imagen
%centroids: son las imágenes centrales de cada clúster
centroids=cen';

indices=[]; %cómo se han reordenado los índices iniciales
matrix=[]; %Número de imágenes en cada clúster (k)

%calcula cuántas imagenes hay en cada clúster
for q=1:k
    count=0;
        for h=1:length(a)
            if class(h) == q
                indices=horzcat(indices,h);
                count=count+1;                
            end           
        end
    matrix=horzcat(matrix,count); 
end
acumulado=cumsum(matrix);

% %HERRAMIENTA VISUAL:
% %Plotea los 25 primeros dígitos de cada clúster
% for t=1:k
%     g=0;
%     if t==1
%         desde = 1;
%         hasta = 25;
%     else
%         desde=acumulado(t-1)+1;
%         hasta=desde+24;
%     end 
%     for w=desde:hasta
%         for i=1:28
%             for j=1:28
%                 digito(i,j)=c((i-1)*28+j,indices(w));
%             end
%         end
%         g=g+1;
%         figure(t)
%         subplot(5,5,g), imshow(digito);
%     end
% end
% 
% %HERRAMIENTA VISUAL:
% %Plotea los los centroides de cada clúster
% for v=1:k
%     for i=1:28
%         for j=1:28
%             digito(i,j)=centroids((i-1)*28+j,v);
%         end
%     end
%     figure(k+1)
%     subplot(4,4,v), imshow(digito);
% end

%Reordenar índices originales
indReordenados=[];
for o=1:length(a)
    new=indices(o);
    original=trainLabels(indices(o));
    indReordenados=horzcat(indReordenados,original);
end

%Ver el índice más común en cada cluster
%Para darle nombre al cluster
centroidLabels=[];
clabel=0;
for t=1:k
    if t==1        
        desde=1;
        hasta=acumulado(1);
    else
        desde=acumulado(t-1)+1;
        hasta=acumulado(t);
    end

    clabel=mode(indReordenados(:,desde:hasta));
    centroidLabels=horzcat(centroidLabels,clabel);

end
%disp('Clúster labels:')
%disp(centroidLabels)

%cálculo error de cada clúster
%ver cuántas etiquetas de cada clúster no coinciden con su nombre
err=[];
for t=1:k
    fallo=0;
    if t==1        
        desde=1;
        hasta=acumulado(1);
    else
        desde=acumulado(t-1)+1;
        hasta=acumulado(t);
    end
    for w=desde:hasta
        if indReordenados(w) ~= centroidLabels(t)
            fallo=fallo+1;
        end    
    end   
    err=horzcat(err,fallo);
end
errorCluster=err./matrix;
%disp('Error de cada clúster:')
%disp(errorCluster)

%cálculo error global
e=cumsum(err);
errorGlobal=e(k)/length(a);
%disp('Error global:')
%disp(errorGlobal)

%Para los datos de TEST:
%Elegir el mejor k
%Guardar los centroids de ese k (savedCent)
%Guardas las labels de los centroids
%Ver la etiqueta de los datos de test (idx_test)
%según el centroido más cercano

save('centroids.mat','centroids');
save('centroidLabels.mat','centroidLabels');

%toc;
end
