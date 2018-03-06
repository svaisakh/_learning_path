function [clusters, centroids] = k_means(X, k)

%%% YOUR CODE HERE

m = size(X,1);
n = size(X,2);

clusters = zeros(m,1);
centroids = X(randperm(m,k),:);

draw_clusters(X,clusters,centroids);
pause;

for iter=1:1000
    oldclusters=clusters;
    for i=1:m
        distances = sum((centroids-X(i,:)).^2,2);
        pos = find(distances==min(distances));
        clusters(i)=pos(1);
    end
    if prod(oldclusters==clusters)
        break;
    end
    draw_clusters(X,clusters,centroids);
    pause(0.01);

    for i=1:k
        pos=find(clusters==i);
        if ~isempty(pos)
            centroids(i,:) = sum(X(pos,:))/length(pos);
        end
    end
    draw_clusters(X,clusters,centroids);
    pause(0.01);
end
% draw_clusters(X,clusters,centroids);