function draw_clusters(X, clusters, centroids)

clf;
hold on;

n = size(X,2);
k = size(centroids,1);

colors=distinguishable_colors(k+1);
colors(4,:)=[];

if n==2
    for i=1:k
        pos = find(clusters==i);
        plot(X(pos,1),X(pos,2),'.','color',colors(i,:));
        plot(centroids(i,1),centroids(i,2),'o','markerfacecolor',colors(i,:),'markersize',10,'linewidth',1.5,'markeredgecolor','k');
    end

    pos = find(clusters==0);
    plot(X(pos,1),X(pos,2),'.','color','k');
    
elseif n==3
    for i=1:k
        pos = find(clusters==i);
        plot3(X(pos,1),X(pos,2),X(pos,3),'.','color',colors(i,:));
        plot3(centroids(i,1),centroids(i,2),centroids(i,3),'o','markerfacecolor',colors(i,:),'markersize',10,'linewidth',1.5,'markeredgecolor','k');
    end

    pos = find(clusters==0);
    plot3(X(pos,1),X(pos,2),X(pos,3),'.','color','k');
    rotate(gca,[1 1 1],45);
end

% % need to actually handle all the different cases due to bug in octave
% if (max(clusters) == 1)
%   plot(X(clusters==1,1), X(clusters==1,2), 'b.');
% elseif (max(clusters) == 2)
%   plot(X(clusters==1,1), X(clusters==1,2), 'b.', ...
%     X(clusters==2,1), X(clusters==2,2), 'g.');
% elseif (max(clusters) == 3)
%   plot(X(clusters==1,1), X(clusters==1,2), 'b.', ...
%     X(clusters==2,1), X(clusters==2,2), 'g.', ...
%     X(clusters==3,1), X(clusters==3,2), 'r.');
% elseif (max(clusters) == 4)
%   plot(X(clusters==1,1), X(clusters==1,2), 'bo', ...
%     X(clusters==2,1), X(clusters==2,2), 'go', ...
%     X(clusters==3,1), X(clusters==3,2), 'ro', ...
%     X(clusters==4,1), X(clusters==4,2), 'co');
% elseif (max(clusters) == 5)
%   plot(X(clusters==1,1), X(clusters==1,2), 'bo', ...
%     X(clusters==2,1), X(clusters==2,2), 'go', ...
%     X(clusters==3,1), X(clusters==3,2), 'ro', ...
%     X(clusters==4,1), X(clusters==4,2), 'co', ...
%     X(clusters==5,1), X(clusters==5,2), 'mo');
% else
%   plot(X(clusters==1,1), X(clusters==1,2), 'bo', ...
%     X(clusters==2,1), X(clusters==2,2), 'go', ...
%     X(clusters==3,1), X(clusters==3,2), 'ro', ...
%     X(clusters==4,1), X(clusters==4,2), 'co', ...
%     X(clusters==5,1), X(clusters==5,2), 'mo', ...
%     X(clusters==6,1), X(clusters==6,2), 'yo');
% end
% 
% plot(centroids(:,1), centroids(:,2), 'kx');
  
