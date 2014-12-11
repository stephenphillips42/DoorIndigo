%% Data visualization
%% Plot by city (first 3 dimentions)
figure
hold on
scatter(Z(:,1).*Z(:,2),Z(:,4).*Z(:,14),50*ones(size(Y_train)),(Y_train),'.');
colorbar
title('First 2 Principal Components')
xlabel('PC1')
ylabel('PC2')
% scatter(Z((Y_city==1),1),Z((Y_city==1),2),1.5.^(Y_train(Y_city==1)),'b.'); 
% scatter(Z((Y_city==2),1),Z((Y_city==2),2),1.5.^(Y_train(Y_city==2)),'r.');
% scatter(Z((Y_city==3),1),Z((Y_city==3),2),1.5.^(Y_train(Y_city==3)),'g.');
% scatter(Z((Y_city==4),1),Z((Y_city==4),2),1.5.^(Y_train(Y_city==4)),'m.');
% scatter(Z((Y_city==5),1),Z((Y_city==5),2),1.5.^(Y_train(Y_city==5)),'k.');
% scatter(Z((Y_city==6),1),Z((Y_city==6),2),1.5.^(Y_train(Y_city==6)),'c.');
% scatter(Z((Y_city==7),1),Z((Y_city==7),2),1.5.^(Y_train(Y_city==7)),'y.')
%%
figure
hold on
miny = prctile(Y_train, 25); % 12
maxy = prctile(Y_train, 75); % 13.566; % Values you want to see between
dimens = [ 1 2 3 ]; % Dimensions to plot

% Interpretation of colors of graph:
% - Blue means between miny < Y < maxy
% - Green means Y > maxy
% - Red means Y < miny
for i = 1:3
    subplot(1,3,i)
plot3(Z((Y_train > miny & Y_train < maxy),dimens(1)),Z((Y_train > miny & Y_train < maxy),dimens(2)),Z((Y_train > miny & Y_train < maxy),dimens(3)),'b.')
hold on
plot3(Z((Y_train > maxy),dimens(1)),Z((Y_train > maxy),dimens(2)),Z((Y_train > maxy),dimens(3)),'g.')
plot3(Z((Y_train < miny),dimens(1)),Z((Y_train < miny),dimens(2)),Z((Y_train < miny),dimens(3)),'r.');
hold off
axis equal
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
title(sprintf('View %d', i))
end

%% See ranges of different dimentions for higher and lower values
figure
hold on
miny = 12;
maxy = 13.566;
dimen = 2;
startdim = 1;
enddim = 500;
rng = enddim-startdim;
for i = startdim:enddim
    plot(3*i+zeros(sum(Y_train > miny & Y_train < maxy),1),Z((Y_train > miny & Y_train < maxy),i),'r.')
    plot(3*i+zeros(sum(Y_train > maxy),1)+1,Z((Y_train > maxy),i),'g.')
    plot(3*i+zeros(sum(Y_train < miny),1)+2,Z((Y_train < miny),i),'b.');
end
%%
figure
cc=hsv(4);
for i = 3
    plot(Z(:,i),Y_train,'.','color',cc(i,:)); hold on;
end
hold off

%%
load('kmeansClusterIds2000.mat')
%clusterIds = labels;
plotpc = [ 1 2 3 ];
nlabels = 4;
cc = hsv(nlabels);
K = 2000;
% clusterData = zeros(K,4);
for i = [ 247 ]%unidrnd(K,1,500)
%     clusterData (i,:) = [i,sum(clusterIds==i),...
%         mean(Y_train(clusterIds==i)),...
%         range(Y_train(clusterIds==i))];
    scatter(Z(clusterIds==i,plotpc(1)),...
          Z(clusterIds==i,plotpc(2)),... Z(clusterIds==i,plotpc(3)),...
          1.5.^(Y_train(clusterIds==i)),...
          (Y_train(clusterIds==i)),...
          '.');
%     plot3(Z(clusterIds==i,plotpc(1)),Z(clusterIds==i,plotpc(2)),Z(clusterIds==i,plotpc(3)),'.','color',cc(i,:));
    hold on
end
colorbar;

title('Zoomed in cluster')
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')


