function plotdata(X,y)

    m = size(X,1);
  
% validate input    

    if(m~=length(y))
        disp('Oops, training set dimension mismatch. Try again');
        return;
    end
    
% bookkeeping
    
    x1 = X(:,1);
    x2 = X(:,2);
    
    h=plot(x1(y==1),x2(y==1),'marker','o','markeredgecolor','k','markerfacecolor','r','markersize',10,'linestyle','none','linewidth',0.75);
    legend(h,'y=1');
    hold on;
    h=plot(x1(y==0),x2(y==0),'marker','o','markeredgecolor','k','markerfacecolor','g','markersize',10,'linestyle','none','linewidth',0.75);
    legend(h,'y=0');
    legend('hide');
    hold off;
end

