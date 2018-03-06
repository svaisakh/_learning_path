function h = wlinreg(X,y,varargin)
    p=inputParser;
    addRequired(p,'X',@isnumeric);
    addRequired(p,'y',@isnumeric);
    addOptional(p,'res',1000,@isnumeric);
    addOptional(p,'color','r',@isstr);
    addOptional(p,'tau',5,@isnumeric);
    parse(p,X,y,varargin{:});
    
    X=p.Results.X;
    y=p.Results.y;
    res=p.Results.res;
    tau=p.Results.tau;
    color=p.Results.color;
    
    xl=xlim;
    
    x = linspace(xl(1),xl(2),res);
    
    for i=2:length(x)
        t=linreg(X,y,weights(x(i),X,tau));
        xspace=linspace(x(i-1),x(i),1000);
        yspace=t(1)+xspace*t(2:end);
        h=plot(xspace,yspace,color);
        if(i~=length(x))
            set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
    end
end

