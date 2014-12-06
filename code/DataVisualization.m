%% Data visualization
%% Plot by city (first 3 dimentions)
figure
plot3(Z((Y_city==1),1),Z((Y_city==1),2),Y_train((Y_city==1)),'b.'); hold on
plot3(Z((Y_city==2),1),Z((Y_city==2),2),Y_train((Y_city==2)),'r.')
plot3(Z((Y_city==3),1),Z((Y_city==3),2),Y_train((Y_city==3)),'g.')
plot3(Z((Y_city==4),1),Z((Y_city==4),2),Y_train((Y_city==4)),'m.')
plot3(Z((Y_city==5),1),Z((Y_city==5),2),Y_train((Y_city==5)),'k.')
%%
figure
hold on
miny = 12;
maxy = 13.566; % Values you want to see between
dimens = [ 1 2 3 ]; % Dimensions to plot

% Interpretation of colors of graph:
% - Blue means between miny < Y < maxy
% - Green means Y > maxy
% - Red means Y < miny

plot3(Z((Y_train > miny & Y_train < maxy),dimens(1)),Z((Y_train > miny & Y_train < maxy),dimens(2)),Z((Y_train > miny & Y_train < maxy),dimens(3)),'b.')
plot3(Z((Y_train > maxy),dimens(1)),Z((Y_train > maxy),dimens(2)),Z((Y_train > maxy),dimens(3)),'g.')
plot3(Z((Y_train < miny),dimens(1)),Z((Y_train < miny),dimens(2)),Z((Y_train < miny),dimens(3)),'r.');


%% See ranges of different dimentions for higher and lower values
figure
hold on
miny = 12;
maxy = 13.566;
dimen = 2;
startdim = 1;
enddim = 75;
rng = enddim-startdim;
for i = startdim:enddim
    plot(3*i+zeros(sum(Y_train > miny & Y_train < maxy),1),Z((Y_train > miny & Y_train < maxy),i),'r.')
    plot(3*i+zeros(sum(Y_train > maxy),1)+1,Z((Y_train > maxy),i),'g.')
    plot(3*i+zeros(sum(Y_train < miny),1)+2,Z((Y_train < miny),i),'b.');
end

%%
clusterIds = labels;
plotpc = [ 7 8 9 ];
nlabels = 4;
cc = hsv(nlabels);
for i = [ 1:4 ]
    fprintf('Cluster %d, with %d members, mean: %f, range %f\n',...
        i,sum(clusterIds==i),...
        mean(Yall(clusterIds==i)),...
        range(Yall(clusterIds==i)))
    scatter3(Z(clusterIds==i,plotpc(1)),...
          Z(clusterIds==i,plotpc(2)),...
          Z(clusterIds==i,plotpc(3)),...
          (Yall(clusterIds==i)).^11/10000000000,...
          100*(clusterIds(clusterIds==i)),...
          '.');
%     plot3(Z(clusterIds==i,plotpc(1)),Z(clusterIds==i,plotpc(2)),Z(clusterIds==i,plotpc(3)),'.','color',cc(i,:));
    hold on
end
colormap;
clear clusterIds;


