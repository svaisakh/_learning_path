function h=plotreg(t,varargin)

% parse all the arguments
    p=inputParser;
    addRequired(p,'t',@isnumeric);
    addParameter(p,'powers',[],@isnumeric);
    parse(p,t,varargin{:});

    powers=p.Results.powers;
    n=2;
    
% validate input    

    if (isempty(powers))
        powers=ones(1,n);
    elseif (length(powers)==1)
        powers=powers*ones(1,n);
    elseif(length(powers)>n)
        disp('Error. More powers asked than number of features!');
    end

    if(sum(powers)+1~=length(t))
        disp('Oops, training set dimension mismatch. Try again');
        return;
    end
    
% bookkeeping
    
    x=sym('x',[n 1]);
    x=enhance(x.',powers);
    x=[sym('c') x];
    d=subs(t'*x.',x(1),1);
    h=fimplicit(d);
    set(h,'color','k');
    legend(h,['Decision boundary [' num2str(powers) ']']);
    legend('hide');
    imagesc(xlim, ylim, decisioncolor(xlim, ylim, 20),'alphadata',0.3);
    colormap(colorGradient([0 1 0],[1 0 0],100));
    
    function C = decisioncolor(xbounds, ybounds, res)
        xspace = linspace(xbounds(1), xbounds(2),res);
        yspace = linspace(ybounds(1), ybounds(2),res);
        
        [X, Y] = meshgrid(xspace,yspace);
        
        C = sigmoid(double(subs(d,{x(2) x(2+powers(1))},{X Y})));
    end
end

